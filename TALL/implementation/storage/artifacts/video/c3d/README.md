Place the pretrained original-style C3D checkpoint for TALL here.

Expected default path:

- `shared://artifacts/video/c3d/c3d.pickle`

The visual feature extraction service reads this location by default. You can override it with:

- `C3D_WEIGHTS_URI`
- `C3D_WEIGHTS_URL`

The checkpoint should contain a PyTorch `state_dict` compatible with the standard Sports1M-style C3D architecture and its `fc6` layer.
