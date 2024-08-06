def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any, Any]:
    d = dict(data)
    output_dir = d.get("output_dir", "")
    output_ext = d.get("output_ext", "")

    os.makedirs(self.output_dir, exist_ok=True)

    img = d.get(self.image_key, None)
    filename = img.meta.get(ImageMetaKey.FILENAME_OR_OBJ) if img is not None else None
    basename = os.path.splitext(os.path.basename(filename))[0] if filename else "mask"
    logger.info(f"File: {filename}; Base: {basename}")

    for key in self.key_iterator(d):
        label = d[key]
        output_filename = f"{basename}{'_' + self.output_postfix if self.output_postfix else ''}{output_ext}"
        output_filepath = os.path.join(output_dir, output_filename)
