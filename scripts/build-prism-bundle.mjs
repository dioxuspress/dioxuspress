// Builds assets/js/prism.bundle.min.js from PrismJS 1.30.0 core + curated
// language components + custom Typst / GPU shader (MSL, CUDA, WGSL) grammars,
// then smoke-tests every language and alias inside node:vm before writing.
//
// Usage:
//   node scripts/build-prism-bundle.mjs
//   node scripts/build-prism-bundle.mjs --cache-dir /path/to/cached/prism
//   node scripts/build-prism-bundle.mjs --out /tmp/prism.bundle.min.js
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createContext, runInContext } from 'node:vm';
import { gzipSync } from 'node:zlib';

const PRISM_VERSION = '1.30.0';
const CDN_BASE = `https://cdn.jsdelivr.net/npm/prismjs@${PRISM_VERSION}`;

// Order matters: components with prerequisites must follow their base grammar
// (e.g. tsx after jsx + typescript; cpp after c; php after markup-templating).
const PRISM_COMPONENTS = [
  'rust',
  'bash',
  'toml',
  'json',
  'yaml',
  'python',
  'swift',
  'jsx',
  'typescript',
  'tsx',
  'markdown',
  'latex',
  'c',
  'cpp',
  'wgsl',
  'sql',
  'diff',
  'go',
  'java',
  'kotlin',
  'csharp',
  'zig',
  'dart',
  'ruby',
  'markup-templating',
  'php',
  'lua',
  'r',
  'docker',
  'protobuf',
  'graphql',
  'makefile',
  'ini',
  'hcl',
];

// Custom Typst grammar (PrismJS 1.30.0 does not ship an official Typst component).
const TYPST_GRAMMAR = `!function(e){var t={comment:[{pattern:/(^|[^\\\\])\\/\\*[\\s\\S]*?(?:\\*\\/|$)/,lookbehind:!0,greedy:!0},{pattern:/(^|[^\\\\:])\\/\\/.*/,lookbehind:!0,greedy:!0}],raw:[{pattern:/\`\`\`[\\s\\S]*?\`\`\`|\`[^\`\\r\\n]*\`/,greedy:!0,alias:"string"}],string:{pattern:/"(?:\\\\[\\s\\S]|[^\\\\"])*"/,greedy:!0},heading:{pattern:/(^[ \\t]*)=+[ \\t].+/m,lookbehind:!0,alias:"important",inside:{punctuation:/^=+/}},math:{pattern:/\\$(?:\\\\[\\s\\S]|[^\\\\$])+\\$/,greedy:!0,alias:"string",inside:{punctuation:/^\\$|\\$$|[(),;]/,function:/#[a-zA-Z_][\\w-]*|\\b(?:frac|sqrt|sum|prod|int|lim|vec|mat|cases|upright|italic|bold|serif|sans|frak|mono|bb|cal)\\b/,symbol:/\\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|infty|nabla|partial|forall|exists|in|notin|subset|subseteq|cup|cap|emptyset|approx|equiv|neq|leq|geq|times|div|pm|mp|cdot|dots)\\b/,operator:/->|=>|<=|>=|!=|==|[-+*/=^_&~']/}},label:{pattern:/<[a-zA-Z_][\\w.-]*:[a-zA-Z_][\\w.-]*>|<[a-zA-Z_][\\w.-]*>/,alias:"symbol"},reference:{pattern:/@[a-zA-Z_][\\w.:-]*/,alias:"atrule"},directive:{pattern:/#(?:let|set|show|import|include|if|else|for|while|in|not|and|or|return|break|continue|context)\\b/,alias:"keyword"},keyword:/\\b(?:let|set|show|import|include|if|else|for|while|in|not|and|or|return|break|continue|context)\\b/,builtin:/\\b(?:page|text|par|heading|list|enum|terms|table|grid|stack|block|box|rect|circle|ellipse|polygon|line|path|curve|place|move|rotate|scale|skew|hide|align|pad|columns|colbreak|pagebreak|v|h|emph|strong|underline|overline|strike|highlight|smallcaps|sub|super|raw|link|cite|footnote|quote|figure|outline|bibliography|counter|state|query|locate|here|measure|layout|style|metadata|image|lorem|rgb|luma|cmyk|oklab|oklch|color|gradient|pattern|stroke|length|angle|ratio|relative|fraction|int|float|str|bytes|label|version|type|content|array|dict|arguments|selector|location|direction|alignment|calc|sys|sym|emoji|pdf|html|math)\\b/,function:{pattern:/#?[a-zA-Z_][\\w-]*(?=\\s*[\\(\\[])|\\.[a-zA-Z_][\\w-]*/,inside:{punctuation:/^[#.]/}},variable:{pattern:/#[a-zA-Z_][\\w-]*/,alias:"variable"},property:{pattern:/\\b[a-zA-Z_][\\w-]*(?=\\s*:(?!:))/,alias:"attr-name"},boolean:/\\b(?:true|false|none|auto|ltr|rtl|ttb|btt|start|end|left|center|right|top|horizon|bottom)\\b/,number:/-?\\b\\d+(?:\\.\\d+)?(?:pt|mm|cm|in|em|deg|rad|%|fr)?\\b|\\b0x[\\da-fA-F]+\\b|\\b0b[01]+\\b|\\b0o[0-7]+\\b/,bold:{pattern:/(^|[^\\w*])\\*[^*\\r\\n]+\\*(?![\\w*])/,lookbehind:!0,greedy:!0,alias:"important"},italic:{pattern:/(^|[^\\w_])_[^_\\r\\n]+_(?![\\w_])/,lookbehind:!0,greedy:!0},operator:/=>|==|!=|<=|>=|\\+=|-=|\\*=|\\/=|&&|\\|\\||[-+*/=<>!~^&|]+/,punctuation:/[{}[\\]();,.:#]/};e.languages.typst=t,e.languages.typ=t}(Prism);`;

// Custom GPU shader extensions: C++11/Metal [[...]] attributes, MSL (metal), CUDA (cu),
// WGSL storage access modes (read / write / read_write), and common fence aliases.
const GPU_AND_ALIAS_EXT = `!function(e){e.languages.insertBefore("cpp","string",{attribute:{pattern:/\\[\\[[^\\[\\]\\r\\n]*\\]\\]/,greedy:!0,inside:{punctuation:/^\\[\\[|\\]\\]$|[(),:]/,number:e.languages.cpp.number,"attr-name":/\\b[a-z_]\\w*\\b/i}}});var m=e.languages.extend("cpp",{keyword:/\\b(?:alignas|alignof|asm|auto|bfloat|bool|break|case|char|class|compl|concept|const|const_cast|constant|constexpr|continue|decltype|default|delete|device|do|double|dynamic_cast|else|enum|explicit|extern|final|float|for|fragment|friend|goto|half|if|inline|int|int16_t|int32_t|int64_t|int8_t|kernel|long|mutable|namespace|new|noexcept|nullptr|object_data|operator|override|private|protected|public|ray_data|register|reinterpret_cast|requires|return|sampler|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread|thread_local|threadgroup|threadgroup_imageblock|typedef|typename|uchar|uint|uint16_t|uint32_t|uint64_t|uint8_t|ulong|union|unsigned|using|vertex|virtual|void|volatile|while)\\b/});e.languages.msl=m,e.languages.insertBefore("msl","string",{"log-banner":{pattern:/^\\[(?:START|END)_KERNEL_COMPILATION\\]$/m,alias:"important"},"fence-marker":{pattern:/^\`\`\`[a-z]*$/m,alias:"comment"}}),e.languages.insertBefore("msl","keyword",{builtin:/\\b(?:(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float)[234]|(?:half|float)[234]x[234]|packed_(?:char|uchar|short|ushort|int|uint|half|float)[234]|atomic_(?:int|uint|ulong|bool|float)|simdgroup_[a-z\\d_]+|(?:texture|depth)(?:1d|2d|3d|cube)(?:_array|_ms)?|f16|bf16|f32|f64|[iu](?:8|16|32|64|size))\\b/,property:{pattern:/(^\\s*)[a-z_]\\w*(?=\\s*:(?!:))/m,lookbehind:!0}}),e.languages.insertBefore("msl","function",{"msl-type":{pattern:/\\b(?:[A-Z]\\w*|(?:float|half|int|uint|short|ushort|char|uchar|bool)_\\d+|\\w+_st)\\b/,alias:"class-name"}}),e.languages.msl["generic-function"]={pattern:/\\b(?!operator\\b|(?:const|dynamic|reinterpret|static)_cast\\b)[a-z_]\\w*\\s*<(?:[^<>]|<[^<>]*>)*>(?=\\s*\\()/i,inside:{function:/^\\w+/,generic:{pattern:/<[\\s\\S]+/,alias:"class-name",inside:e.languages.msl}}},e.languages.metal=e.languages.msl;var c=e.languages.extend("cpp",{keyword:/\\b(?:__constant__|__device__|__forceinline__|__global__|__grid_constant__|__host__|__launch_bounds__|__managed__|__noinline__|__restrict__|__shared__|alignas|alignof|asm|auto|bool|break|case|catch|char|class|const|const_cast|constexpr|continue|decltype|default|delete|do|double|dynamic_cast|else|enum|explicit|extern|final|float|for|friend|goto|if|inline|int|int16_t|int32_t|int64_t|int8_t|long|mutable|namespace|new|noexcept|nullptr|operator|override|private|protected|public|register|reinterpret_cast|return|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread_local|throw|try|typedef|typename|uint16_t|uint32_t|uint64_t|uint8_t|union|unsigned|using|virtual|void|volatile|while)\\b/});e.languages.cuda=c,e.languages.insertBefore("cuda","keyword",{builtin:/\\b(?:blockDim|blockIdx|gridDim|threadIdx|warpSize|__syncthreads|__syncwarp|atomicAdd|atomicCAS|atomicExch|atomicMax|atomicMin|atomicSub|(?:char|uchar|short|ushort|int|uint|long|ulong|longlong|ulonglong|float|double)[1234]|dim3|half|__half|__nv_bfloat16)\\b/,"cuda-launch":{pattern:/<<<[^<>\\r\\n]*>>>/,inside:{punctuation:/^<<<|>>>$|,/,rest:null}},property:{pattern:/(^\\s*)[a-z_]\\w*(?=\\s*:(?!:))/m,lookbehind:!0}}),e.languages.insertBefore("cuda","function",{"cuda-type":{pattern:/\\b[A-Z]\\w*\\b/,alias:"class-name"}}),e.languages.cuda["generic-function"]={pattern:/\\b(?!operator\\b|(?:const|dynamic|reinterpret|static)_cast\\b)[a-z_]\\w*\\s*<(?:[^<>]|<[^<>]*>)*>(?=\\s*\\()/i,inside:{function:/^\\w+/,generic:{pattern:/<[\\s\\S]+/,alias:"class-name",inside:e.languages.cuda}}},e.languages.cuda["cuda-launch"].inside.rest=e.languages.cuda,e.languages.cu=e.languages.cuda,e.languages.wgsl&&(e.languages.insertBefore("wgsl","comment",{"fence-marker":{pattern:/^\`\`\`[a-z]*$/m,alias:"comment"},property:{pattern:/(^\\s*)source(?=\\s*:(?!:))/m,lookbehind:!0}}),e.languages.wgsl.keyword=/\\b(?:bitcast|break|case|const|continue|continuing|default|discard|else|enable|fallthrough|fn|for|function|if|let|loop|private|read_write|read|return|storage|struct|switch|type|uniform|var|while|workgroup|write)\\b/);var l=e.languages;l.rs=l.rust,l.zsh=l.bash,l.cc=l.cpp,l.cxx=l.cpp,l.hpp=l.cpp,l.golang=l.go,l.proto=l.protobuf,l.gql=l.graphql,l.make=l.makefile,l.terraform=l.hcl,l.tf=l.hcl,l.mdx=l.markdown}(Prism);`;

const parseArgs = (argv) => {
  const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
  let cacheDir = null;
  let outFile = join(repoRoot, 'assets/js/prism.bundle.min.js');
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--cache-dir' && argv[i + 1]) {
      cacheDir = argv[i + 1];
      i += 1;
    } else if (argv[i] === '--out' && argv[i + 1]) {
      outFile = argv[i + 1];
      i += 1;
    }
  }
  return { cacheDir, outFile };
};

const loadSource = async (cacheDir, fileName, remotePath) => {
  if (cacheDir) {
    const cached = join(cacheDir, fileName);
    if (existsSync(cached)) {
      return readFileSync(cached, 'utf8').trim();
    }
  }
  const url = `${CDN_BASE}/${remotePath}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  }
  const text = (await res.text()).trim();
  if (cacheDir) {
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(join(cacheDir, fileName), `${text}\n`);
  }
  return text;
};

const stripSourceMapComment = (code) =>
  code.replace(/\r?\n?\/\/# sourceMappingURL=.*$/m, '').trim();

const verifyBundle = (bundle) => {
  const ctx = { Prism: { manual: true } };
  ctx.window = ctx;
  ctx.self = ctx;
  ctx.global = ctx;
  createContext(ctx);
  runInContext(bundle, ctx);

  const { Prism } = ctx;
  const samples = [
    ['markup', '<div class="app">Hello</div>', ['html', 'svg', 'mathml']],
    ['xml', '<note><to>User</to></note>', ['rss', 'atom', 'ssml']],
    ['css', '.btn { color: #ff6600; }', []],
    ['javascript', 'const add = (a, b) => a + b;', ['js']],
    ['jsx', 'const el = <Button label={name} />;', []],
    ['typescript', 'interface User { id: number; }', ['ts']],
    ['tsx', 'const View = ({ id }: Props) => <span>{id}</span>;', []],
    ['rust', 'pub fn main() -> Result<(), Error> { Ok(()) }', ['rs']],
    ['bash', 'export RUST_LOG=info && cargo test', ['sh', 'shell', 'zsh']],
    ['toml', '[package]\nname = "dioxuspress"', []],
    ['json', '{"name": "dioxuspress", "ok": true}', ['webmanifest']],
    ['yaml', 'site:\n  title: DioxusPress', ['yml']],
    ['python', 'def greet(name: str) -> str:\n  return f"hi {name}"', ['py']],
    ['swift', 'func greet(_ name: String) -> String { "hi \\(name)" }', []],
    ['markdown', '# Heading\n\n**bold** and `code`', ['md', 'mdx']],
    ['latex', '\\section{Intro} $\\alpha + \\beta$', ['tex', 'context']],
    ['typst', '#set page(paper: "a4")\n= Heading', ['typ']],
    ['c', '#include <stdio.h>\nint main(void) { return 0; }', []],
    ['cpp', 'template <typename T> [[nodiscard]] T id(T x) { return x; }', ['cc', 'cxx', 'hpp']],
    ['msl', '[[kernel]] void saxpy(device float4* x [[buffer(0)]]) {}', ['metal']],
    ['cuda', '__global__ void k(float* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; }', ['cu']],
    ['wgsl', '@group(0) @binding(0) var<storage, read_write> buf: array<f32>;', []],
    ['sql', 'SELECT id, title FROM posts WHERE published = TRUE;', []],
    ['diff', '--- a/lib.rs\n+++ b/lib.rs\n@@ -1 +1 @@\n-old\n+new', []],
    ['go', 'package main\nfunc main() { go work() }', ['golang']],
    ['java', 'public record Point(int x, int y) {}', []],
    ['kotlin', 'suspend fun fetch(id: Long): Result<User> = runCatching { api(id) }', ['kt', 'kts']],
    ['csharp', 'public sealed record Order(Guid Id, decimal Total);', ['cs', 'dotnet']],
    ['zig', 'const std = @import("std");\npub fn main() !void {}', []],
    ['dart', 'Future<void> main() async { final x = 42; }', []],
    ['ruby', 'class Post < ApplicationRecord\n  validates :title, presence: true\nend', ['rb']],
    ['php', '<?php declare(strict_types=1); fn(int $x): int => $x * 2;', []],
    ['lua', 'local function fib(n) if n < 2 then return n end return fib(n - 1) end', []],
    ['r', 'df <- data.frame(x = 1:5, y = rnorm(5))', []],
    ['docker', 'FROM rust:1-slim AS builder\nWORKDIR /app\nRUN cargo build --release', ['dockerfile']],
    ['protobuf', 'syntax = "proto3";\nmessage Ping { int64 id = 1; }', ['proto']],
    ['graphql', 'type Query { post(slug: ID!): Post }', ['gql']],
    ['makefile', '.PHONY: build\nbuild:\n\tcargo build --release', ['make']],
    ['ini', '[database]\nhost = 127.0.0.1\nport = 5432', []],
    ['hcl', 'resource "aws_s3_bucket" "assets" {\n  bucket = "dioxuspress"\n}', ['terraform', 'tf']],
  ];

  for (const [lang, code, aliases] of samples) {
    const grammar = Prism.languages[lang];
    if (!grammar) {
      throw new Error(`Missing Prism grammar: ${lang}`);
    }
    const html = Prism.highlight(code, grammar, lang);
    if (!html.includes('class="token ')) {
      throw new Error(`Language "${lang}" produced zero highlight tokens`);
    }
    for (const alias of aliases) {
      if (Prism.languages[alias] !== grammar) {
        throw new Error(`Alias "${alias}" does not point to "${lang}"`);
      }
    }
  }
  return samples.length;
};

const main = async () => {
  const { cacheDir, outFile } = parseArgs(process.argv.slice(2));
  const core = stripSourceMapComment(await loadSource(cacheDir, 'prism.min.js', 'prism.min.js'));

  const parts = [
    `/* PrismJS ${PRISM_VERSION} core + curated language bundle (generated by scripts/build-prism-bundle.mjs).\n   Run \`node scripts/build-prism-bundle.mjs\` after editing the language list. */`,
    `${core};`,
  ];

  for (const name of PRISM_COMPONENTS) {
    const raw = await loadSource(cacheDir, `prism-${name}.min.js`, `components/prism-${name}.min.js`);
    parts.push(`${stripSourceMapComment(raw)};`);
    if (name === 'latex') {
      parts.push(TYPST_GRAMMAR);
    }
  }
  parts.push(GPU_AND_ALIAS_EXT);

  const bundle = `${parts.join('\n')}\n`;
  const count = verifyBundle(bundle);
  writeFileSync(outFile, bundle);

  const rawBytes = Buffer.byteLength(bundle);
  const gzBytes = gzipSync(bundle).length;
  console.log(
    `build-prism-bundle: verified ${count} languages -> ${outFile} (${rawBytes} B raw, ${gzBytes} B gzip)`,
  );
};

await main();
