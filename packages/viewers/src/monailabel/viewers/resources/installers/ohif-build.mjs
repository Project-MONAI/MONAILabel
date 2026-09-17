/** Build a pinned OHIF checkout with the MONAI Label extension. Node 22+, Git, Corepack.
 * Called as: node ohif-build.mjs SOURCE EXTENSION_RESOURCES
 * No server imports, shell snippets, platform-specific setup scripts, or user prompt execution.
 */
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const recipe = JSON.parse(
  fs.readFileSync(path.join(here, "ohif.json"), "utf8"),
);
const [sourceArg, resourcesArg] = process.argv.slice(2);
if (!sourceArg || !resourcesArg)
  throw new Error("Expected SOURCE and EXTENSION_RESOURCES paths.");
if (Number(process.versions.node.split(".")[0]) < recipe.node_major)
  throw new Error(`OHIF build requires Node ${recipe.node_major} or later.`);
const source = path.resolve(sourceArg),
  resources = path.resolve(resourcesArg);
const env = {
  ...process.env,
  APP_CONFIG: recipe.app_config,
  PUBLIC_URL: recipe.public_url,
  NODE_OPTIONS: "--max-old-space-size=12288",
};
const run = (exe, args, cwd = source) =>
  execFileSync(exe, args, { cwd, env, stdio: "inherit" });

// Invoke Corepack's JS entry point using Node, avoiding Windows .cmd/shell quoting.
function corepackEntry() {
  const dirs = [
    path.dirname(process.execPath),
    ...(process.env.PATH || "").split(path.delimiter),
  ];
  for (const dir of dirs) {
    const candidates = [
      path.join(dir, "corepack"),
      path.join(dir, "node_modules/corepack/dist/corepack.js"),
      path.resolve(dir, "../lib/node_modules/corepack/dist/corepack.js"),
    ];
    for (const candidate of candidates) {
      if (!fs.existsSync(candidate)) continue;
      const resolved = fs.realpathSync(candidate);
      if (resolved.endsWith(".js")) return resolved;
    }
  }
  throw new Error(
    "Corepack is required to build OHIF. Install it alongside Node.",
  );
}
const corepack = corepackEntry();
if (!fs.existsSync(source)) {
  fs.mkdirSync(path.dirname(source), { recursive: true });
  run(
    "git",
    [
      "clone",
      "--depth",
      "1",
      "--branch",
      "v" + recipe.version,
      recipe.repository,
      source,
    ],
    path.dirname(source),
  );
}
const head = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: source,
  encoding: "utf8",
}).trim();
if (head !== recipe.commit)
  throw new Error("Cached OHIF checkout differs from the pinned release.");
if (!fs.existsSync(path.join(source, "node_modules/webpack")))
  run(process.execPath, [
    corepack,
    "yarn",
    "install",
    "--frozen-lockfile",
    "--non-interactive",
  ]);
// Webpack folds ONNX's typeof __filename test but leaves its Node variable
// unbound in this browser entry. Remove only that Node-specific fallback; a
// cached chunk has no document.currentScript and otherwise crashes on reload.
const onnxEntry = path.join(
  source,
  "node_modules/onnxruntime-web/dist/esm/ort.webgpu.min.js",
);
const onnxCode = fs.readFileSync(onnxEntry, "utf8");
const nodeFallback = /typeof __filename<"u"&&\((\w+)=\1\|\|__filename\)/g;
const browserFallback = "/* monailabel browser filename */void 0";
const matches = [...onnxCode.matchAll(nodeFallback)];
if (matches.length === 4) {
  fs.writeFileSync(onnxEntry, onnxCode.replace(nodeFallback, browserFallback));
  // Webpack treats node_modules as immutable, so invalidate its cached bundle.
  fs.rmSync(path.join(source, "platform/app/node_modules/.cache/webpack"), {
    recursive: true,
    force: true,
  });
} else if (!onnxCode.includes(browserFallback)) {
  throw new Error(
    "Pinned ONNX browser bundle changed; review its filename fallback.",
  );
}
const configPath = path.join(source, "platform/app/pluginConfig.json");
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
for (const [folder, category, name] of [
  ["extension", "extensions", "@monailabel/extension"],
  ["mode", "modes", "@monailabel/mode"],
]) {
  const destination = path.join(source, category, "monailabel");
  fs.cpSync(path.join(resources, folder), destination, { recursive: true });
  const link = path.join(source, "node_modules", name);
  fs.mkdirSync(path.dirname(link), { recursive: true });
  if (!fs.existsSync(link))
    fs.symlinkSync(
      destination,
      link,
      process.platform === "win32" ? "junction" : "dir",
    );
  if (!config[category].some((item) => item.packageName === name))
    config[category].push({ packageName: name });
}
fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n");
fs.copyFileSync(
  path.join(resources, "config.js"),
  path.join(source, "platform/app/public/config/monailabel.js"),
);
run(process.execPath, [
  corepack,
  "yarn",
  "workspace",
  "@ohif/app",
  "run",
  "build:viewer",
]);
const dist = path.join(source, "platform/app/dist");
const entry = path.join(dist, "index.html");
if (!fs.existsSync(entry))
  throw new Error("OHIF build did not produce its entry page.");
// Externalize the upstream inline bootstraps so the server can retain its strict script policy.
const html = fs
  .readFileSync(entry, "utf8")
  .replace(/<script>(.*?)<\/script>/gs, (_, script) => {
    const name =
      "bootstrap-" +
      createHash("sha256").update(script).digest("hex").slice(0, 16) +
      ".js";
    fs.writeFileSync(path.join(dist, name), script);
    return `<script src="${recipe.public_url}${name}"></script>`;
  });
fs.writeFileSync(entry, html);
console.log("OHIF build ready:", dist);
