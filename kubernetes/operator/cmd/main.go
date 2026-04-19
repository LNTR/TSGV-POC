package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const (
	managedBy                 = "model-router-operator"
	routerConfigSuffix        = "active-routing"
	groupName                 = "registry.tsgv.io"
	groupVersion              = "v1alpha1"
	kindName                  = "ModelRouter"
	routePlural               = "modelrouters"
	routeNameLabel            = "registry.tsgv.io/router-name"
	activeModelAnnotation     = "registry.tsgv.io/active-model"
	targetServiceAnnotation   = "registry.tsgv.io/target-service"
	targetNamespaceAnnotation = "registry.tsgv.io/target-namespace"
	compatibleModelsAnnot     = "registry.tsgv.io/compatible-models"
	sharedAnnot               = "registry.tsgv.io/shared"
)

var (
	scheme     = runtime.NewScheme()
	routerGVK  = schema.GroupVersionKind{Group: groupName, Version: groupVersion, Kind: kindName}
	errBadSpec = errors.New("invalid model router spec")
)

type routeSpec struct {
	ServiceName        string   `json:"serviceName"`
	Namespace          string   `json:"namespace,omitempty"`
	Port               int32    `json:"port"`
	ImplementationName string   `json:"implementationName,omitempty"`
	CompatibleModels   []string `json:"compatibleModels,omitempty"`
	Shared             bool     `json:"shared,omitempty"`
}

type reconciledRoute struct {
	ServiceName        string   `json:"serviceName"`
	Namespace          string   `json:"namespace"`
	Port               int32    `json:"port"`
	ImplementationName string   `json:"implementationName"`
	CompatibleModels   []string `json:"compatibleModels"`
	Shared             bool     `json:"shared"`
}

type modelSpec struct {
	Services map[string]routeSpec `json:"services"`
}

type routerSpec struct {
	ActiveModel string               `json:"activeModel"`
	Models      map[string]modelSpec `json:"models"`
}

type routerReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
}

func main() {
	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

	manager, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		log.Fatalf("create manager: %v", err)
	}

	router := &unstructured.Unstructured{}
	router.SetGroupVersionKind(routerGVK)

	if err := ctrl.NewControllerManagedBy(manager).
		For(router).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Complete(&routerReconciler{
			Client: manager.GetClient(),
			Scheme: manager.GetScheme(),
		}); err != nil {
		log.Fatalf("create controller: %v", err)
	}

	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		log.Fatalf("start manager: %v", err)
	}
}

func (r *routerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := logf.FromContext(ctx).WithValues("modelRouter", req.NamespacedName.String())
	router := &unstructured.Unstructured{}
	router.SetGroupVersionKind(routerGVK)

	if err := r.Get(ctx, req.NamespacedName, router); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	activeModel, routes, err := normalizeActiveRoutes(req.Namespace, router.Object)
	if err != nil {
		setErr := r.updateStatus(ctx, router, map[string]any{
			"observedGeneration": router.GetGeneration(),
			"lastReconciledAt":   time.Now().UTC().Format(time.RFC3339),
			"error":              err.Error(),
		})
		if setErr != nil {
			logger.Error(setErr, "update status for invalid spec")
		}
		return ctrl.Result{}, nil
	}

	for logicalService, route := range routes {
		if err := r.ensureAliasService(ctx, router, activeModel, logicalService, route); err != nil {
			return ctrl.Result{}, err
		}
	}

	if err := r.cleanupStaleAliases(ctx, req.Namespace, req.Name, routeKeys(routes)); err != nil {
		return ctrl.Result{}, err
	}

	configMapName, err := r.ensureRouterConfigMap(ctx, router, activeModel, routes)
	if err != nil {
		return ctrl.Result{}, err
	}

	if err := writeRoutesToMongoDB(ctx, req.Namespace, req.Name, activeModel, routes); err != nil {
		return ctrl.Result{}, err
	}

	status := map[string]any{
		"observedGeneration": router.GetGeneration(),
		"activeModel":        activeModel,
		"managedAliases":     routeKeys(routes),
		"configMap":          configMapName,
		"lastReconciledAt":   time.Now().UTC().Format(time.RFC3339),
	}
	if err := r.updateStatus(ctx, router, status); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("reconciled model router", "activeModel", activeModel, "aliases", len(routes))
	return ctrl.Result{}, nil
}

func normalizeActiveRoutes(namespace string, object map[string]any) (string, map[string]reconciledRoute, error) {
	specValue, found, err := unstructured.NestedMap(object, "spec")
	if err != nil || !found {
		return "", nil, fmt.Errorf("%w: spec must be set", errBadSpec)
	}

	payload, err := json.Marshal(specValue)
	if err != nil {
		return "", nil, fmt.Errorf("marshal spec: %w", err)
	}

	var spec routerSpec
	if err := json.Unmarshal(payload, &spec); err != nil {
		return "", nil, fmt.Errorf("%w: decode spec: %v", errBadSpec, err)
	}

	activeModel := strings.TrimSpace(spec.ActiveModel)
	if activeModel == "" {
		return "", nil, fmt.Errorf("%w: spec.activeModel must be set", errBadSpec)
	}

	activeModelSpec, ok := spec.Models[activeModel]
	if !ok {
		return "", nil, fmt.Errorf("%w: active model %q is not defined under spec.models", errBadSpec, activeModel)
	}

	if len(activeModelSpec.Services) == 0 {
		return "", nil, fmt.Errorf("%w: model %q must define at least one logical service", errBadSpec, activeModel)
	}

	normalized := make(map[string]reconciledRoute, len(activeModelSpec.Services))
	for logicalService, rawRoute := range activeModelSpec.Services {
		serviceName := strings.TrimSpace(rawRoute.ServiceName)
		if serviceName == "" {
			return "", nil, fmt.Errorf("%w: service %q must define serviceName", errBadSpec, logicalService)
		}
		if rawRoute.Port <= 0 {
			return "", nil, fmt.Errorf("%w: service %q must define a positive port", errBadSpec, logicalService)
		}

		routeNamespace := strings.TrimSpace(rawRoute.Namespace)
		if routeNamespace == "" {
			routeNamespace = namespace
		}

		compatibleModels := cleanedCompatibleModels(rawRoute.CompatibleModels, activeModel)
		implementationName := strings.TrimSpace(rawRoute.ImplementationName)
		if implementationName == "" {
			implementationName = fmt.Sprintf("%s-%s", activeModel, logicalService)
		}

		normalized[logicalService] = reconciledRoute{
			ServiceName:        serviceName,
			Namespace:          routeNamespace,
			Port:               rawRoute.Port,
			ImplementationName: implementationName,
			CompatibleModels:   compatibleModels,
			Shared:             rawRoute.Shared,
		}
	}

	return activeModel, normalized, nil
}

func cleanedCompatibleModels(models []string, activeModel string) []string {
	if len(models) == 0 {
		return []string{activeModel}
	}

	seen := make(map[string]struct{}, len(models))
	cleaned := make([]string, 0, len(models))
	for _, model := range models {
		trimmed := strings.TrimSpace(model)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		cleaned = append(cleaned, trimmed)
	}

	if len(cleaned) == 0 {
		return []string{activeModel}
	}

	return cleaned
}

func (r *routerReconciler) ensureAliasService(
	ctx context.Context,
	router *unstructured.Unstructured,
	activeModel string,
	logicalService string,
	route reconciledRoute,
) error {
	service := &corev1.Service{}
	key := types.NamespacedName{Namespace: router.GetNamespace(), Name: logicalService}
	err := r.Get(ctx, key, service)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	desired := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      logicalService,
			Namespace: router.GetNamespace(),
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": managedBy,
				routeNameLabel:                 router.GetName(),
			},
			Annotations: map[string]string{
				activeModelAnnotation:     activeModel,
				targetServiceAnnotation:   route.ServiceName,
				targetNamespaceAnnotation: route.Namespace,
				compatibleModelsAnnot:     strings.Join(route.CompatibleModels, ","),
				sharedAnnot:               strconv.FormatBool(route.Shared),
			},
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: buildExternalName(route),
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       route.Port,
					TargetPort: intstrFromInt32(route.Port),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	if err := controllerutil.SetControllerReference(router, desired, r.Scheme); err != nil {
		return err
	}

	if apierrors.IsNotFound(err) {
		return r.Create(ctx, desired)
	}

	if service.Labels["app.kubernetes.io/managed-by"] != managedBy {
		return fmt.Errorf("service %q already exists and is not managed by %s", logicalService, managedBy)
	}

	service.Labels = desired.Labels
	service.Annotations = desired.Annotations
	service.OwnerReferences = desired.OwnerReferences
	service.Spec.Type = desired.Spec.Type
	service.Spec.ExternalName = desired.Spec.ExternalName
	service.Spec.Ports = desired.Spec.Ports
	service.Spec.Selector = nil
	service.Spec.ClusterIP = ""
	service.Spec.ClusterIPs = nil
	service.Spec.IPFamilies = nil
	service.Spec.IPFamilyPolicy = nil
	service.Spec.SessionAffinity = ""
	return r.Update(ctx, service)
}

func (r *routerReconciler) cleanupStaleAliases(
	ctx context.Context,
	namespace string,
	routerName string,
	activeAliases []string,
) error {
	serviceList := &corev1.ServiceList{}
	if err := r.List(ctx, serviceList,
		client.InNamespace(namespace),
		client.MatchingLabels{
			"app.kubernetes.io/managed-by": managedBy,
			routeNameLabel:                 routerName,
		},
	); err != nil {
		return err
	}

	for _, service := range serviceList.Items {
		if slices.Contains(activeAliases, service.Name) {
			continue
		}
		if err := r.Delete(ctx, service.DeepCopy()); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

func (r *routerReconciler) ensureRouterConfigMap(
	ctx context.Context,
	router *unstructured.Unstructured,
	activeModel string,
	routes map[string]reconciledRoute,
) (string, error) {
	configMapName := fmt.Sprintf("%s-%s", router.GetName(), routerConfigSuffix)
	payloadRoutes := make(map[string]map[string]any, len(routes))
	for logicalService, route := range routes {
		payloadRoutes[logicalService] = map[string]any{
			"serviceName":        route.ServiceName,
			"namespace":          route.Namespace,
			"port":               route.Port,
			"compatibleModels":   route.CompatibleModels,
			"shared":             route.Shared,
			"implementationName": route.ImplementationName,
			"aliasHost":          logicalService,
			"aliasURL":           fmt.Sprintf("http://%s:%d", logicalService, route.Port),
			"externalName":       buildExternalName(route),
		}
	}

	payload := map[string]any{
		"routerName":  router.GetName(),
		"activeModel": activeModel,
		"generatedAt": time.Now().UTC().Format(time.RFC3339),
		"routes":      payloadRoutes,
	}

	payloadBytes, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal routing payload: %w", err)
	}

	configMap := &corev1.ConfigMap{}
	key := types.NamespacedName{Namespace: router.GetNamespace(), Name: configMapName}
	err = r.Get(ctx, key, configMap)
	if err != nil && !apierrors.IsNotFound(err) {
		return "", err
	}

	desired := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: router.GetNamespace(),
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": managedBy,
				routeNameLabel:                 router.GetName(),
			},
		},
		Data: map[string]string{
			"activeModel": activeModel,
			"routes.json": string(payloadBytes),
		},
	}

	if err := controllerutil.SetControllerReference(router, desired, r.Scheme); err != nil {
		return "", err
	}

	if apierrors.IsNotFound(err) {
		if err := r.Create(ctx, desired); err != nil {
			return "", err
		}
		return configMapName, nil
	}

	configMap.Labels = desired.Labels
	configMap.OwnerReferences = desired.OwnerReferences
	configMap.Data = desired.Data
	if err := r.Update(ctx, configMap); err != nil {
		return "", err
	}

	return configMapName, nil
}

func (r *routerReconciler) updateStatus(ctx context.Context, router *unstructured.Unstructured, status map[string]any) error {
	updated := router.DeepCopy()
	jsonCompatibleStatus, err := normalizeJSONMap(status)
	if err != nil {
		return err
	}
	if err := unstructured.SetNestedMap(updated.Object, jsonCompatibleStatus, "status"); err != nil {
		return err
	}
	return r.Status().Update(ctx, updated)
}

func normalizeJSONMap(payload map[string]any) (map[string]any, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal status payload: %w", err)
	}

	normalized := make(map[string]any)
	if err := json.Unmarshal(raw, &normalized); err != nil {
		return nil, fmt.Errorf("unmarshal status payload: %w", err)
	}
	return normalized, nil
}

func writeRoutesToMongoDB(
	ctx context.Context,
	namespace string,
	routerName string,
	activeModel string,
	routes map[string]reconciledRoute,
) error {
	mongoURI := strings.TrimSpace(os.Getenv("MONGODB_URI"))
	if mongoURI == "" {
		return nil
	}

	databaseName := valueOrDefault(os.Getenv("MODEL_ROUTER_MONGODB_DATABASE"), "service_registry")
	summaryCollectionName := valueOrDefault(os.Getenv("MODEL_ROUTER_SUMMARY_COLLECTION"), "model_router_state")
	routesCollectionName := valueOrDefault(os.Getenv("MODEL_ROUTER_ROUTES_COLLECTION"), "routes")
	timeoutMS := valueOrDefault(os.Getenv("MODEL_ROUTER_MONGODB_TIMEOUT_MS"), "3000")

	timeoutValue, err := strconv.Atoi(timeoutMS)
	if err != nil {
		return fmt.Errorf("parse MODEL_ROUTER_MONGODB_TIMEOUT_MS: %w", err)
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutValue)*time.Millisecond)
	defer cancel()

	mongoClient, err := mongo.Connect(timeoutCtx, options.Client().ApplyURI(mongoURI))
	if err != nil {
		return fmt.Errorf("connect mongodb: %w", err)
	}
	defer func() {
		disconnectCtx, disconnectCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer disconnectCancel()
		if err := mongoClient.Disconnect(disconnectCtx); err != nil {
			log.Printf("disconnect mongodb: %v", err)
		}
	}()

	database := mongoClient.Database(databaseName)
	summaryCollection := database.Collection(summaryCollectionName)
	routesCollection := database.Collection(routesCollectionName)

	indexCtx, indexCancel := context.WithTimeout(context.Background(), time.Duration(timeoutValue)*time.Millisecond)
	defer indexCancel()

	if _, err := summaryCollection.Indexes().CreateOne(indexCtx, mongo.IndexModel{
		Keys:    bson.D{{Key: "namespace", Value: 1}, {Key: "router_name", Value: 1}},
		Options: options.Index().SetUnique(true),
	}); err != nil {
		return fmt.Errorf("create summary index: %w", err)
	}

	if _, err := routesCollection.Indexes().CreateOne(indexCtx, mongo.IndexModel{
		Keys: bson.D{
			{Key: "namespace", Value: 1},
			{Key: "router_name", Value: 1},
			{Key: "logical_service", Value: 1},
		},
		Options: options.Index().SetUnique(true),
	}); err != nil {
		return fmt.Errorf("create routes index: %w", err)
	}

	now := time.Now().UTC().Format(time.RFC3339)
	if _, err := summaryCollection.UpdateOne(
		timeoutCtx,
		bson.M{"namespace": namespace, "router_name": routerName},
		bson.M{
			"$set": bson.M{
				"active_model": activeModel,
				"updated_at":   now,
				"routes":       routes,
			},
			"$setOnInsert": bson.M{"created_at": now},
		},
		options.Update().SetUpsert(true),
	); err != nil {
		return fmt.Errorf("update summary document: %w", err)
	}

	for logicalService, route := range routes {
		if _, err := routesCollection.UpdateOne(
			timeoutCtx,
			bson.M{
				"namespace":       namespace,
				"router_name":     routerName,
				"logical_service": logicalService,
			},
			bson.M{
				"$set": bson.M{
					"active_model":        activeModel,
					"implementation_name": route.ImplementationName,
					"target_service":      route.ServiceName,
					"target_namespace":    route.Namespace,
					"target_port":         route.Port,
					"compatible_models":   route.CompatibleModels,
					"shared":              route.Shared,
					"external_name":       buildExternalName(route),
					"alias_host":          logicalService,
					"alias_url":           fmt.Sprintf("http://%s:%d", logicalService, route.Port),
					"updated_at":          now,
				},
				"$setOnInsert": bson.M{"created_at": now},
			},
			options.Update().SetUpsert(true),
		); err != nil {
			return fmt.Errorf("update route %q: %w", logicalService, err)
		}
	}

	return nil
}

func routeKeys(routes map[string]reconciledRoute) []string {
	keys := make([]string, 0, len(routes))
	for key := range routes {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func buildExternalName(route reconciledRoute) string {
	return fmt.Sprintf("%s.%s.svc.cluster.local", route.ServiceName, route.Namespace)
}

func valueOrDefault(value string, fallback string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fallback
	}
	return trimmed
}

func intstrFromInt32(value int32) intstr.IntOrString {
	return intstr.FromInt32(value)
}
