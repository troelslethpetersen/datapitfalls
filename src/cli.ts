#!/usr/bin/env node
// datapitfalls — command-line entry point

import { DOMAINS, TAGLINE, VERSION, ruleCount, ruleCountsByDomain } from './index.js';
import { detectPitfalls } from './analyze.js';
import { PROVIDER_DEFAULTS, inferProviderFromModel } from './providers/index.js';
import type { ProviderName } from './providers/index.js';
import { formatReport, hasBlockingFindings } from './report.js';
import { buildScanInput, buildChainInput } from './scan-input.js';

const useColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;

// Powered By Data palette, as 24-bit truecolor SGR parameters.
const LEMON = '38;2;226;229;35'; // Electric Lemon #E2E523
const OCEAN = '38;2;31;134;182'; // Ocean Blue     #1F86B6
const SKY = '38;2;105;223;250'; // Sky Blue       #69DFFA
const IRON = '38;2;60;55;68'; // Iron Gray      #3C3744
const RESET_CODE = '\x1b[0m';

const paint = (code: string, s: string): string => (useColor && code ? `\x1b[${code}m${s}${RESET_CODE}` : s);
const dim = (s: string): string => (useColor ? `\x1b[2m${s}${RESET_CODE}` : s);

// Accent roles per terminal background. Body text is never colored — it uses the
// terminal's own foreground, so it stays readable on light or dark. A dark terminal
// gets the bright brand colors; a light one, the darker members of the palette.
interface Theme {
  face: string;
  bevel: string;
  heading: string;
  label: string;
  command: string;
  hint: string;
  border: string;
}
const DARK_THEME: Theme = {
  face: LEMON,
  bevel: OCEAN,
  heading: `1;${SKY}`,
  label: `1;${LEMON}`,
  command: SKY,
  hint: OCEAN,
  border: OCEAN,
};
const LIGHT_THEME: Theme = {
  face: OCEAN,
  bevel: IRON,
  heading: `1;${OCEAN}`,
  label: `1;${OCEAN}`,
  command: OCEAN,
  hint: IRON,
  border: IRON,
};

// "DATA" stacked over "PITFALLS" — figlet "ANSI Shadow", colored two-tone for a
// retro extruded look (bright face + dark bevel). Rendered once and embedded.
const WORDMARK = [
  '██████╗  █████╗ ████████╗ █████╗ ',
  '██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗',
  '██║  ██║███████║   ██║   ███████║',
  '██║  ██║██╔══██║   ██║   ██╔══██║',
  '██████╔╝██║  ██║   ██║   ██║  ██║',
  '╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝',
  '██████╗ ██╗████████╗███████╗ █████╗ ██╗     ██╗     ███████╗',
  '██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗██║     ██║     ██╔════╝',
  '██████╔╝██║   ██║   █████╗  ███████║██║     ██║     ███████╗',
  '██╔═══╝ ██║   ██║   ██╔══╝  ██╔══██║██║     ██║     ╚════██║',
  '██║     ██║   ██║   ██║     ██║  ██║███████╗███████╗███████║',
  '╚═╝     ╚═╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝',
];

// Two-tone the block art: solid █ faces in one color, the ╗╝║═ bevel edges in
// another, so the letters look extruded.
function colorizeWordmark(line: string, face: string, bevel: string): string {
  if (!useColor) return line;
  let out = '';
  let cur = '';
  for (const ch of line) {
    if (ch === ' ') {
      if (cur !== '') {
        out += RESET_CODE;
        cur = '';
      }
      out += ' ';
      continue;
    }
    const want = ch === '█' ? face : bevel;
    if (want !== cur) {
      out += `\x1b[${want}m`;
      cur = want;
    }
    out += ch;
  }
  return cur !== '' ? out + RESET_CODE : out;
}

function introBox(border: string): string {
  const width = 70;
  const lines = [
    'Catch data pitfalls in any data work — a chart, a code file, a',
    'report, or a description — whether a person or an AI produced it.',
    'Get back what is wrong, why it matters, and how to fix it.',
  ];
  const horiz = '─'.repeat(width - 2);
  const edge = (s: string): string => paint(border, s);
  const body = lines.map((l) => '  ' + edge('│') + ' ' + l.padEnd(width - 4) + ' ' + edge('│'));
  return ['  ' + edge('╭' + horiz + '╮'), ...body, '  ' + edge('╰' + horiz + '╯')].join('\n');
}

function printSplash(theme: Theme): void {
  console.log();
  for (const line of WORDMARK) console.log('  ' + colorizeWordmark(line, theme.face, theme.bevel));
  console.log();
  console.log('  Check the data work of humans and AI alike for the pitfalls that mislead.');
  console.log();
  console.log(introBox(theme.border));
  console.log();
  console.log('  ' + paint(theme.heading, 'Getting Started:'));
  console.log(
    '    ' + paint(theme.label, 'Human:') + '  ' + paint(theme.command, 'datapitfalls scan <file>') + '  scan a chart, code, report, or description'
  );
  console.log('            ' + paint(theme.hint, 'add --all for every finding, --thorough for the deepest model'));
  console.log(
    '    ' + paint(theme.label, 'Agent:') + '  ' + paint(theme.command, 'datapitfalls scan --json <file>') + '  machine-readable findings'
  );
  console.log('            ' + paint(theme.hint, 'add --ci to exit non-zero when a blocking pitfall is found'));
  console.log();
  console.log('  ' + paint(theme.hint, `Run datapitfalls --help for the full command list · v${VERSION}`));
  console.log('  ' + dim('Made by Data Literacy · github.com/bjonesdataliteracy/datapitfalls'));
  console.log();
}

// Best-effort terminal-background detection so the splash adapts. Order: explicit
// DATAPITFALLS_THEME override, the COLORFGBG env var, then an OSC 11 query with a
// short timeout. Falls back to the dark theme.
type ThemeName = 'light' | 'dark';

function queryBackgroundColor(): Promise<ThemeName> {
  return new Promise((resolve) => {
    const stdin = process.stdin;
    let buf = '';
    let settled = false;
    const finish = (name: ThemeName): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stdin.removeListener('data', onData);
      try {
        stdin.setRawMode(false);
      } catch {
        /* not a raw-capable TTY */
      }
      stdin.pause();
      resolve(name);
    };
    const onData = (chunk: Buffer): void => {
      buf += chunk.toString('latin1');
      const m = buf.match(/rgb:([0-9a-fA-F]{2,4})\/([0-9a-fA-F]{2,4})\/([0-9a-fA-F]{2,4})/);
      if (!m) return;
      const [, r, g, b] = m;
      if (r === undefined || g === undefined || b === undefined) return;
      const hi = (h: string): number => parseInt(h.slice(0, 2), 16);
      const luminance = (0.2126 * hi(r) + 0.7152 * hi(g) + 0.0722 * hi(b)) / 255;
      finish(luminance > 0.5 ? 'light' : 'dark');
    };
    const timer = setTimeout(() => finish('dark'), 150);
    try {
      stdin.setRawMode(true);
      stdin.resume();
      stdin.on('data', onData);
      process.stdout.write('\x1b]11;?\x07');
    } catch {
      finish('dark');
    }
  });
}

async function detectTheme(explicit?: ThemeName): Promise<Theme> {
  const override = explicit ?? process.env.DATAPITFALLS_THEME;
  let name: ThemeName | undefined =
    override === 'light' || override === 'dark' ? override : undefined;
  if (!name && process.env.COLORFGBG) {
    const bg = Number(process.env.COLORFGBG.split(';').pop());
    if (!Number.isNaN(bg)) name = bg === 7 || bg === 15 ? 'light' : 'dark';
  }
  if (!name) name = useColor && process.stdin.isTTY ? await queryBackgroundColor() : 'dark';
  return name === 'light' ? LIGHT_THEME : DARK_THEME;
}

function printStats(): void {
  const counts = ruleCountsByDomain();
  console.log(`datapitfalls v${VERSION} — ${ruleCount()} rules across ${DOMAINS.length} domains\n`);
  for (const domain of DOMAINS) {
    console.log(`  ${String(counts[domain]).padStart(3)}  ${domain}`);
  }
}

function printHelp(): void {
  console.log(
    `datapitfalls v${VERSION} — ${TAGLINE}\n` +
      'Usage:\n' +
      '  datapitfalls stats               Show the pitfall catalog size by domain\n' +
      '  datapitfalls scan <file>         Scan a code file, analysis description, chart image, or PDF for pitfalls\n' +
      '  datapitfalls scan <a.png> <b.png> …  Scan several charts together (cross-chart pitfalls)\n' +
      '    --text                         Treat the file as a plain-English analysis description\n' +
      '    --provider <name>              Which LLM provider to use: anthropic (default), openai, or gemini\n' +
      '    --model <id>                   Override the model id (provider is inferred if not set)\n' +
      '    --thorough                     Use the provider\'s deepest model (Opus 4.7 / GPT-5 / Gemini 2.5 Pro)\n' +
      '    --fast                         Use the provider\'s cheapest model (Haiku 4.5 / GPT-5 mini / Gemini 2.5 Flash)\n' +
      '    --all                          Show all findings, incl. lower-confidence latent ones\n' +
      '    --summary                      Lead with an overall summary, consequence ratings, and avoided pitfalls\n' +
      '    --json                         Output the full report as JSON\n' +
      '    --ci                           Exit non-zero if an active error/warning is found\n' +
      '\nImage files (.png/.jpg/.jpeg/.gif/.webp) are scanned with the provider\'s vision model;\n' +
      'pass several to scan them as a set. PDFs (.pdf) are read as native documents (prose +\n' +
      'charts/tables) by every provider, Word docs (.docx) as prose, slide decks (.pptx) as\n' +
      'per-slide text + charts, and notebooks (.ipynb) as their extracted code.\n' +
      '\nThe scan command needs an API key for the chosen provider:\n' +
      '  anthropic → ANTHROPIC_API_KEY   (default)\n' +
      '  openai    → OPENAI_API_KEY\n' +
      '  gemini    → GOOGLE_API_KEY or GEMINI_API_KEY\n' +
      'Model defaults are claude-sonnet-4-6 / gpt-5 / gemini-2.5-pro; override with --model or\n' +
      'the ANTHROPIC_MODEL / OPENAI_MODEL / GEMINI_MODEL env vars.\n' +
      '\nThe splash adapts to your terminal background; force it with --theme <light|dark>\n' +
      'or the DATAPITFALLS_THEME env var.'
  );
}

const PROVIDERS: ProviderName[] = ['anthropic', 'openai', 'gemini'];

function envKeyFor(provider: ProviderName): string[] {
  switch (provider) {
    case 'anthropic':
      return ['ANTHROPIC_API_KEY'];
    case 'openai':
      return ['OPENAI_API_KEY'];
    case 'gemini':
      return ['GOOGLE_API_KEY', 'GEMINI_API_KEY'];
  }
}

async function scan(args: string[]): Promise<void> {
  const files: string[] = [];
  let modelOverride: string | undefined;
  let providerOverride: ProviderName | undefined;
  let speed: 'fast' | 'thorough' | undefined;
  let showAll = false;
  let asJson = false;
  let ci = false;
  let forceText = false;
  let chain = false;
  let summary = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === undefined) continue;
    if (arg === '--thorough') {
      speed = 'thorough';
    } else if (arg === '--fast') {
      speed = 'fast';
    } else if (arg === '--all') {
      showAll = true;
    } else if (arg === '--summary') {
      summary = true;
    } else if (arg === '--json') {
      asJson = true;
    } else if (arg === '--ci') {
      ci = true;
    } else if (arg === '--text') {
      forceText = true;
    } else if (arg === '--chain') {
      chain = true;
    } else if (arg === '--provider' || arg.startsWith('--provider=')) {
      const value = arg === '--provider' ? args[++i] : arg.slice('--provider='.length);
      if (!value || !(PROVIDERS as string[]).includes(value)) {
        console.error(`--provider must be one of: ${PROVIDERS.join(', ')}`);
        process.exitCode = 1;
        return;
      }
      providerOverride = value as ProviderName;
    } else if (arg === '--model' || arg.startsWith('--model=')) {
      const value = arg === '--model' ? args[++i] : arg.slice('--model='.length);
      if (!value) {
        console.error('--model requires a model id (e.g. gpt-5, gemini-2.5-pro).');
        process.exitCode = 1;
        return;
      }
      modelOverride = value;
    } else if (arg.startsWith('-')) {
      console.error(`Unknown option: ${arg}`);
      process.exitCode = 1;
      return;
    } else {
      files.push(arg);
    }
  }

  if (files.length === 0) {
    console.error(
      'Usage: datapitfalls scan [--provider anthropic|openai|gemini] [--model <id>] [--thorough|--fast] [--chain] <file> [more files…]'
    );
    process.exitCode = 1;
    return;
  }

  // Resolve provider: explicit > inferred from model > default Anthropic.
  const provider: ProviderName =
    providerOverride ?? inferProviderFromModel(modelOverride) ?? 'anthropic';

  // Resolve model: explicit --model > --fast/--thorough mapped to the provider's
  // model family > provider default (engine picks it up if we leave undefined).
  let model = modelOverride;
  if (!model && speed) {
    model = PROVIDER_DEFAULTS[provider][speed];
  }

  const keys = envKeyFor(provider);
  if (!keys.some((k) => process.env[k])) {
    console.error(
      `${keys.join(' / ')} is not set. Export your ${provider} API key to run a scan.`
    );
    process.exitCode = 1;
    return;
  }

  const result = chain ? await buildChainInput(files) : await buildScanInput(files, forceText);
  if ('error' in result) {
    console.error(result.error);
    process.exitCode = 1;
    return;
  }

  const report = await detectPitfalls(result.input, {
    provider,
    ...(model ? { model } : {}),
    ...(summary ? { variant: 'summary' as const } : {}),
  });
  console.log(asJson ? JSON.stringify(report, null, 2) : formatReport(report, { showAll }));

  if (ci && hasBlockingFindings(report)) process.exitCode = 1;
}

/** Pull a `--theme light|dark` (or `--theme=light`) flag out of argv, returning it
 *  plus the remaining args. The splash is the only themed output. */
function extractThemeFlag(argv: string[]): { theme?: ThemeName; rest: string[] } {
  const rest: string[] = [];
  let theme: ThemeName | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--theme') {
      const value = argv[i + 1];
      if (value === 'light' || value === 'dark') {
        theme = value;
        i += 1;
      }
      continue;
    }
    if (arg !== undefined && arg.startsWith('--theme=')) {
      const value = arg.slice('--theme='.length);
      if (value === 'light' || value === 'dark') theme = value;
      continue;
    }
    if (arg !== undefined) rest.push(arg);
  }
  return { theme, rest };
}

async function main(): Promise<void> {
  const { theme, rest } = extractThemeFlag(process.argv.slice(2));
  const [command, ...args] = rest;

  switch (command) {
    case undefined:
      printSplash(await detectTheme(theme));
      break;
    case 'stats':
      printStats();
      break;
    case 'scan':
      await scan(args);
      break;
    case 'help':
    case '--help':
    case '-h':
      printHelp();
      break;
    default:
      printHelp();
  }
}

main().catch((error: unknown) => {
  console.error(`datapitfalls error: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
