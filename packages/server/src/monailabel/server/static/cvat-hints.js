/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Source-frame prompts are viewer state, never CVAT annotation objects.
export function createHints(canvas) {
  const ns = "http://www.w3.org/2000/svg";
  let options = {},
    stop = () => {},
    change = () => {},
    group,
    content,
    drag;
  const element = (tag, attributes) => {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes))
      node.setAttribute(key, value);
    return node;
  };
  function point(event) {
    const matrix = content.getScreenCTM();
    if (!matrix) return null;
    const svg = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    const { offset } = canvas().geometry;
    const [x, y] = [svg.x - offset, svg.y - offset];
    if (!drag && (x < 0 || y < 0 || x >= options.width || y >= options.height))
      return null;
    return [
      Math.min(options.width - 1, Math.max(0, x)),
      Math.min(options.height - 1, Math.max(0, y)),
    ];
  }
  function publish(hints) {
    options = { ...options, hints };
    change(hints);
    render();
  }
  function down(event) {
    if (
      options.disabled ||
      !options.mode ||
      options.mode === "navigate" ||
      event.button !== 0
    )
      return;
    if (event.target.closest("[data-prompt-index]")) return;
    const p = point(event);
    if (!p) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (options.mode === "box") {
      drag = { point: p, key: options.key, pointer: event.pointerId };
      content.setPointerCapture(event.pointerId);
    } else {
      if ((options.hints || []).filter((h) => h.kind === "point").length >= 64)
        return;
      publish([
        ...(options.hints || []),
        {
          kind: "point",
          positive: options.mode === "positive",
          coordinates: [p],
        },
      ]);
    }
  }
  function up(event) {
    if (!drag) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const start = drag;
    drag = null;
    if (content.hasPointerCapture(start.pointer))
      content.releasePointerCapture(start.pointer);
    if (start.key !== options.key || options.disabled) return;
    const end = point(event);
    if (
      !end ||
      Math.abs(end[0] - start.point[0]) < 1 ||
      Math.abs(end[1] - start.point[1]) < 1
    )
      return;
    const corners = [
      start.point.map((v, i) => Math.min(v, end[i])),
      start.point.map((v, i) => Math.max(v, end[i])),
    ];
    publish([
      ...(options.hints || []).filter((h) => h.kind !== "box"),
      { kind: "box", coordinates: corners },
    ]);
  }
  function render() {
    const svg = canvas()?.html().querySelector("#cvat_canvas_content");
    if (!svg) return;
    if (svg !== content) {
      detach();
      content = svg;
      content.addEventListener("pointerdown", down, true);
      content.addEventListener("pointerup", up, true);
      content.addEventListener("pointercancel", cancel, true);
      document.addEventListener("keydown", escape, true);
    }
    group?.remove();
    group = element("g", {
      class: "monailabel-prompts",
      "pointer-events": "none",
    });
    const { offset, scale } = canvas().geometry;
    (options.hints || []).forEach((hint, index) => {
      const p = hint.coordinates[0];
      const shape =
        hint.kind === "point"
          ? element("circle", {
              cx: p[0] + offset,
              cy: p[1] + offset,
              r: 5 / scale,
              fill: hint.positive ? "#16a34a" : "#dc2626",
              stroke: "white",
            })
          : element("rect", {
              x: p[0] + offset,
              y: p[1] + offset,
              width: hint.coordinates[1][0] - p[0],
              height: hint.coordinates[1][1] - p[1],
              fill: "none",
              stroke: "#22c8df",
              "stroke-dasharray": "5 3",
            });
      shape.setAttribute("stroke-width", "2");
      shape.setAttribute("vector-effect", "non-scaling-stroke");
      shape.setAttribute("data-prompt-index", index);
      shape.setAttribute("pointer-events", options.disabled ? "none" : "all");
      shape.style.cursor = "pointer";
      const title = element("title", {});
      title.textContent = "Remove this input";
      shape.append(title);
      shape.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopImmediatePropagation();
        if (!options.disabled)
          publish(options.hints.filter((_, i) => i !== index));
      });
      group.append(shape);
    });
    content.append(group);
  }
  function escape(event) {
    if (event.key === "Escape") {
      cancel();
      stop();
    }
  }
  function cancel() {
    drag = null;
  }
  function detach() {
    content?.removeEventListener("pointerdown", down, true);
    content?.removeEventListener("pointerup", up, true);
    content?.removeEventListener("pointercancel", cancel, true);
    document.removeEventListener("keydown", escape, true);
    group?.remove();
    content = null;
    drag = null;
  }
  return {
    configure(next, onChange, onStop) {
      stop = onStop;
      if (options.key !== next.key || options.mode !== next.mode) cancel();
      options = next;
      change = onChange;
      render();
    },
    destroy: detach,
  };
}
