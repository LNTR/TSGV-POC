FROM golang:1.25 AS builder

WORKDIR /workspace

COPY go.mod ./go.mod
COPY go.sum ./go.sum
RUN go mod download

COPY cmd ./cmd
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/model-router-operator ./cmd

FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /
COPY --from=builder /out/model-router-operator /model-router-operator

ENTRYPOINT ["/model-router-operator"]
