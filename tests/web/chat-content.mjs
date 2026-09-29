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

import assert from "node:assert/strict";
import test from "node:test";
import { renderMarkdown } from "../../packages/server/src/monailabel/server/static/chat-content.js";

test("assistant Markdown preserves tables, paragraphs, lists, line breaks and code", () => {
  const html = renderMarkdown(
    "**Training defaults**\n\n| Parameter | Default |\n| --- | --- |\n| Epochs | 5 |\n\n" +
      "- First\n- Second\n\nLine one\nLine two\n\n```python\nprint('<test>')\n```",
  );
  for (const tag of [
    "strong",
    "table",
    "thead",
    "tbody",
    "ul",
    "li",
    "pre",
    "code",
  ])
    assert.match(html, new RegExp(`<${tag}[ >]`));
  assert.match(html, /Line one<br>\nLine two/);
  assert.match(html, /&lt;test&gt;/);
});

test("untrusted replies cannot execute HTML or scripts or embed remote images", () => {
  const html = renderMarkdown(
    '<script>alert(1)</script>\n<img src=x onerror="alert(1)">\n\n' +
      "[bad](javascript:alert%281%29) [bad](data:text/html,boom) " +
      "![tracking](https://example.org/tracker.png)\n\n[Docs](https://example.org/docs)",
  );
  assert.doesNotMatch(html, /<(script|img)|href="(?:javascript|data):/);
  assert.match(
    html,
    /href="https:\/\/example.org\/docs" target="_blank" rel="noopener noreferrer"/,
  );
});
