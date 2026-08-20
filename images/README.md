# Images

Place external handwritten digit images in this folder.

Recommended filename:

```text
modified.png
```

The prediction script can also accept a custom image path:

```bash
python predict_external.py path/to/your/image.png
```

The image is converted to grayscale, optionally inverted, resized to `28×28`,
normalized to `0–1`, flattened into `784` features, and passed to the KNN model.
