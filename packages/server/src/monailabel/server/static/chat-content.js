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

import MarkdownIt from "./vendor/markdown-it-15.0.2.min.mjs";

const markdown = new MarkdownIt({ html: false, breaks: true, linkify: false });
// Replies can contain untrusted model/tool text. Never embed remote images or HTML.
markdown.disable("image");
const openLink = markdown.renderer.rules.link_open;
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  tokens[index].attrSet("target", "_blank");
  tokens[index].attrSet("rel", "noopener noreferrer");
  return openLink
    ? openLink(tokens, index, options, env, renderer)
    : renderer.renderToken(tokens, index, options);
};

export const renderMarkdown = (text) => markdown.render(String(text ?? ""));

export function chatContent(text, user = false) {
  const node = document.createElement("div");
  node.className = "chat-content";
  if (user) {
    node.classList.add("chat-plain");
    node.textContent = text;
    return node;
  }
  node.innerHTML = renderMarkdown(text);
  for (const table of node.querySelectorAll("table")) {
    const headings = [...table.querySelectorAll("thead th")].map(
      (cell) => cell.textContent,
    );
    if (headings.length > 2) table.classList.add("chat-wide-table");
    for (const row of table.querySelectorAll("tbody tr")) {
      [...row.cells].forEach((cell, index) => {
        cell.dataset.label = headings[index] || "";
      });
    }
  }
  return node;
}
