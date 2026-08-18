import React, { useEffect, useRef, useState, useCallback } from "react";

/* ----------------------------------------------------------------- motion */

const reduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Fires once when the element first enters the viewport.
 *
 * Motion here is decoration, never a gate on content, so this deliberately has
 * three independent paths to the visible state: IntersectionObserver, a
 * geometry check on mount, and a rAF-throttled scroll fallback. If observers
 * are unavailable or never fire, the content still shows.
 */
export function useInView({ threshold = 0.15, rootMargin = "0px 0px -8% 0px" } = {}) {
  const ref = useRef(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      setInView(true);
      return undefined;
    }

    let done = false;
    let frame = 0;

    const show = () => {
      if (done) return;
      done = true;
      setInView(true);
      cleanup();
    };

    // margin mirrors the observer's rootMargin closely enough for the fallback
    const visible = () => {
      const r = node.getBoundingClientRect();
      const h = window.innerHeight || document.documentElement.clientHeight;
      return r.top < h * 0.92 && r.bottom > 0;
    };

    const check = () => {
      frame = 0;
      if (visible()) show();
    };

    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(check);
    };

    let observer;
    if (typeof IntersectionObserver !== "undefined") {
      observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) show();
        },
        { threshold, rootMargin }
      );
      observer.observe(node);
    }

    let poll;
    function cleanup() {
      if (observer) observer.disconnect();
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) cancelAnimationFrame(frame);
      if (poll) clearInterval(poll);
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    check();

    // Low-frequency geometry poll. This covers the case where the observer and
    // scroll events are both dead, without the blind timeout an earlier version
    // used: that fired while the element was still far below the fold, so the
    // animation had already finished by the time it was scrolled to.
    poll = setInterval(check, 700);

    return () => {
      clearInterval(poll);
      cleanup();
    };
  }, [threshold, rootMargin]);

  return [ref, inView];
}

export function Reveal({ children, className = "", delay = 0, from = "up" }) {
  const [ref, inView] = useInView();
  const offset = { up: "translate-y-4", left: "-translate-x-4", right: "translate-x-4", none: "" }[from];

  return (
    <div
      ref={ref}
      className={`${className} transition-all duration-[720ms] ease-[cubic-bezier(0.16,1,0.3,1)] ${
        inView ? "translate-x-0 translate-y-0 opacity-100 blur-0" : `${offset} opacity-0 blur-[2px]`
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/** Counts up to `value` the first time it is seen. */
export function CountUp({ value, decimals = 0, duration = 1400, prefix = "", suffix = "" }) {
  const [ref, inView] = useInView({ threshold: 0.4 });
  const [display, setDisplay] = useState(0);
  const frame = useRef();

  useEffect(() => {
    if (!inView) return undefined;
    if (reduced()) {
      setDisplay(value);
      return undefined;
    }
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      // ease-out cubic keeps the last digits from crawling
      setDisplay(value * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);

    // Frames can be suspended (background tab, reduced power). Timers still
    // run, so settle on the true figure regardless. Displaying a stalled 0
    // would misreport the data, which matters more than the animation.
    const settle = setTimeout(() => setDisplay(value), duration + 700);

    return () => {
      cancelAnimationFrame(frame.current);
      clearTimeout(settle);
    };
  }, [inView, value, duration]);

  const shown = decimals
    ? display.toFixed(decimals)
    : Math.round(display).toLocaleString("en-US");

  return (
    <span ref={ref} className="tabular-nums">
      {prefix}
      {shown}
      {suffix}
    </span>
  );
}

/**
 * Kinetic typography: sets a line word by word as it is reached.
 *
 * Each word carries its own delay so the phrase assembles rather than fading
 * in as a block. The text is real text in the DOM throughout, so it stays
 * selectable and readable to assistive tech; only the transform is staged.
 */
export function Kinetic({
  children,
  as: Tag = "span",
  className = "",
  delay = 0,
  stagger = 55,
  from = "up",
}) {
  const [ref, inView] = useInView({ threshold: 0.2 });
  const words = String(children).split(" ");

  const offset = {
    up: "translateY(0.5em)",
    down: "translateY(-0.4em)",
    scale: "scale(0.94)",
  }[from];

  return (
    <Tag ref={ref} className={className}>
      {words.map((word, i) => (
        <span key={`${word}-${i}`} className="inline-block overflow-hidden align-bottom">
          <span
            className="inline-block will-change-transform"
            style={{
              opacity: inView ? 1 : 0,
              transform: inView ? "none" : offset,
              filter: inView ? "blur(0)" : "blur(3px)",
              transition:
                "opacity 620ms cubic-bezier(0.16,1,0.3,1), transform 620ms cubic-bezier(0.16,1,0.3,1), filter 620ms ease-out",
              transitionDelay: `${delay + i * stagger}ms`,
            }}
          >
            {word}
          </span>
          {i < words.length - 1 && " "}
        </span>
      ))}
    </Tag>
  );
}

/**
 * Decorative Nastaliq set large and faint, used to give typographically thin
 * sections some presence. Always aria-hidden: it is texture, not content.
 */
export function Nastaliq({ children, className = "", size = "18rem", drift = true }) {
  const [ref, inView] = useInView({ threshold: 0.05 });
  return (
    <span
      ref={ref}
      aria-hidden="true"
      className={`pointer-events-none absolute select-none font-urdu leading-none ${className}`}
      style={{
        fontSize: size,
        direction: "rtl",
        opacity: inView ? 1 : 0,
        transform: inView || !drift ? "translateX(0)" : "translateX(1.5rem)",
        transition: "opacity 1400ms ease-out, transform 1600ms cubic-bezier(0.16,1,0.3,1)",
      }}
    >
      {children}
    </span>
  );
}

/* ----------------------------------------------------------------- layout */

export function Section({
  id,
  index,
  eyebrow,
  title,
  lead,
  children,
  tone = "ivory",
  className = "",
}) {
  const tones = {
    ivory: "bg-ivory",
    cream: "bg-cream",
    forest: "bg-forest text-sand",
  };
  const dark = tone === "forest";

  return (
    <section id={id} className={`relative overflow-hidden ${tones[tone]} ${className}`}>
      <div className="relative mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-28">
        {(eyebrow || title || lead) && (
          <header className="relative mb-12 max-w-3xl">
            {index && (
              <Reveal from="none">
                <span
                  aria-hidden="true"
                  className={`pointer-events-none absolute -top-10 right-0 select-none font-serif text-[5.5rem] leading-none sm:-top-14 sm:text-[8rem] ${
                    dark ? "text-white/[0.06]" : "text-forest/[0.055]"
                  }`}
                >
                  {index}
                </span>
              </Reveal>
            )}
            {eyebrow && (
              <Reveal from="none">
                <p className={dark ? "eyebrow text-sand-deep" : "eyebrow"}>{eyebrow}</p>
              </Reveal>
            )}
            {title && (
              <Kinetic
                as="h2"
                delay={60}
                className={`mt-3 text-3xl leading-tight sm:text-4xl ${dark ? "!text-cream" : ""}`}
              >
                {title}
              </Kinetic>
            )}
            {lead && (
              <Reveal delay={120}>
                <p
                  className={`mt-5 text-base leading-relaxed sm:text-lg ${
                    dark ? "text-sand/85" : "text-forest-mid"
                  }`}
                >
                  {lead}
                </p>
              </Reveal>
            )}
          </header>
        )}
        {children}
      </div>
    </section>
  );
}

export function Prose({ children, className = "" }) {
  return (
    <div className={`max-w-prose space-y-5 text-[0.975rem] leading-[1.75] text-forest-mid ${className}`}>
      {children}
    </div>
  );
}

/** Marks where a number came from, so no figure on the page is unattributed. */
export function Source({ children, className = "" }) {
  return (
    <p className={`mt-3 font-sans text-xs text-forest-soft ${className}`}>
      <span className="font-medium">Source:</span> {children}
    </p>
  );
}

/**
 * lucide-react 1.x no longer ships brand marks, so the GitHub glyph is inlined.
 * Takes the same `size` prop as a lucide icon and inherits currentColor.
 */
export function GithubMark({ size = 16, className = "", ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
      {...rest}
    >
      <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58l-.01-2.05c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.2.08 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.34-5.47-5.96 0-1.32.47-2.39 1.24-3.23-.12-.31-.54-1.53.12-3.18 0 0 1.01-.32 3.3 1.23a11.4 11.4 0 0 1 6.01 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.65.24 2.87.12 3.18.77.84 1.24 1.91 1.24 3.23 0 4.63-2.81 5.65-5.49 5.95.43.37.82 1.1.82 2.22l-.01 3.29c0 .32.21.7.82.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5Z" />
    </svg>
  );
}

/* ----------------------------------------------------------------- pieces */

export function StatCard({ value, label, note, tone = "light" }) {
  const dark = tone === "dark";
  return (
    <div
      className={`group rounded-lg border p-5 transition-all duration-300 hover:-translate-y-0.5 ${
        dark
          ? "border-sand/25 bg-white/5 hover:border-sand/45"
          : "border-sand-deep/60 bg-white/70 hover:border-forest-soft/60 hover:shadow-[0_8px_24px_-16px_rgba(38,52,43,0.45)]"
      }`}
    >
      <div
        className={`font-serif text-2xl leading-none sm:text-[1.75rem] ${
          dark ? "text-cream" : "text-forest-deep"
        }`}
      >
        {value}
      </div>
      <div className={`mt-2 text-sm font-medium ${dark ? "text-sand" : "text-forest"}`}>{label}</div>
      {note && (
        <div className={`mt-1 text-xs leading-relaxed ${dark ? "text-sand/60" : "text-forest-soft"}`}>
          {note}
        </div>
      )}
    </div>
  );
}

export function Pill({ children, tone = "sand" }) {
  const tones = {
    sand: "bg-sand/60 text-forest border-sand-deep/60",
    merlot: "bg-merlot-wash text-merlot-mid border-merlot-bright/30",
    forest: "bg-forest/8 text-forest-mid border-forest/15",
    outline: "bg-transparent text-forest-soft border-sand-deep",
    glow: "bg-merlot text-cream border-merlot",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Callout({ icon: Icon, title, children, tone = "sand" }) {
  const tones = {
    sand: "border-sand-deep bg-sand/35",
    merlot: "border-merlot-bright/35 bg-merlot-wash/70",
  };
  return (
    <div
      className={`rounded-lg border-l-2 border-y border-r px-5 py-4 transition-colors ${tones[tone]}`}
    >
      <div className="flex items-start gap-3">
        {Icon && <Icon size={17} className="mt-0.5 shrink-0 text-merlot-mid" aria-hidden="true" />}
        <div>
          {title && <p className="font-sans text-sm font-semibold text-forest-deep">{title}</p>}
          <div className="mt-1 text-sm leading-relaxed text-forest-mid">{children}</div>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- tables */

/**
 * The first child is the header row's cells (pass them in a fragment); every
 * remaining child becomes a body row. Tables size to their container rather
 * than forcing a horizontal scroll, so cells wrap on narrow screens.
 */
export function Table({ children, caption, className = "" }) {
  const items = Array.isArray(children) ? children : [children];
  const [head, ...body] = items;

  return (
    <div className={`rounded-lg border border-sand-deep/60 bg-white/60 ${className}`}>
      <table className="w-full table-auto border-collapse text-[0.8rem] sm:text-sm">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-sand-deep/70 bg-sand/30">{head}</tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
  );
}

export function Th({ children, align = "left", className = "" }) {
  return (
    <th
      scope="col"
      className={`px-2.5 py-3 sm:px-4 text-${align} font-sans text-[0.62rem] font-semibold uppercase leading-tight tracking-wider text-forest-soft sm:text-[0.7rem] ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({ children, align = "left", strong = false, className = "" }) {
  return (
    <td
      className={`px-2.5 py-3 sm:px-4 text-${align} ${
        strong ? "font-medium text-forest-deep" : "text-forest-mid"
      } ${className}`}
    >
      {children}
    </td>
  );
}

export function Tr({ children, highlight = false }) {
  return (
    <tr
      className={`border-b border-sand-deep/40 transition-colors last:border-0 ${
        highlight ? "bg-merlot-wash/45" : "hover:bg-sand/25"
      }`}
    >
      {children}
    </tr>
  );
}

/* ----------------------------------------------------------------- charts */

const fmt = new Intl.NumberFormat("en-US");
export const n = (v) => fmt.format(v);

/** Horizontal bars, filling from zero the first time they are seen. */
export function BarChart({ data, max, unit = "", tone = "forest", showValue = true }) {
  const [ref, inView] = useInView({ threshold: 0.25 });
  const [hover, setHover] = useState(null);
  const peak = max ?? Math.max(...data.map((d) => d.value));
  const tones = { forest: "bg-forest", merlot: "bg-merlot-mid", mixed: null };

  return (
    <ul ref={ref} className="space-y-3" onMouseLeave={() => setHover(null)}>
      {data.map((d, i) => (
        <li
          key={d.label}
          className="group transition-opacity duration-200"
          onMouseEnter={() => setHover(i)}
          style={{ opacity: hover === null || hover === i ? 1 : 0.45 }}
        >
          <div className="flex items-baseline justify-between gap-4 text-sm">
            <span className="font-medium text-forest-deep">{d.label}</span>
            {showValue && (
              <span className="shrink-0 font-mono text-xs tabular-nums text-forest-mid">
                {typeof d.display === "string" ? (
                  d.display
                ) : (
                  <CountUp value={d.value} duration={1000} />
                )}
                {unit}
              </span>
            )}
          </div>
          <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-sand/70">
            <div
              className={`h-full rounded-full transition-[width] duration-[900ms] ease-[cubic-bezier(0.16,1,0.3,1)] ${
                d.tone ? tones[d.tone] : tones[tone]
              }`}
              style={{
                width: inView ? `${peak ? (d.value / peak) * 100 : 0}%` : "0%",
                transitionDelay: `${i * 70}ms`,
              }}
            />
          </div>
          {d.note && <p className="mt-1 text-xs text-forest-soft">{d.note}</p>}
        </li>
      ))}
    </ul>
  );
}

/** Donut for a part-to-whole split, drawing itself in on first view. */
export function Donut({ data, size = 190, thickness = 26, centerLabel, centerValue }) {
  const [ref, inView] = useInView({ threshold: 0.3 });
  const total = data.reduce((s, d) => s + d.value, 0);
  const radius = (size - thickness) / 2;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <div ref={ref} className="flex flex-wrap items-center gap-7">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={centerLabel}
        className="shrink-0"
      >
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          {data.map((d, i) => {
            const length = total ? (d.value / total) * circumference : 0;
            const el = (
              <circle
                key={d.label}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke={d.color}
                strokeWidth={thickness}
                strokeDasharray={
                  inView ? `${length} ${circumference - length}` : `0 ${circumference}`
                }
                strokeDashoffset={-offset}
                style={{
                  transition: "stroke-dasharray 900ms cubic-bezier(0.16,1,0.3,1)",
                  transitionDelay: `${i * 120}ms`,
                }}
              />
            );
            offset += length;
            return el;
          })}
        </g>
        <text
          x="50%"
          y="47%"
          textAnchor="middle"
          className="fill-forest-deep font-serif"
          style={{ fontSize: 22 }}
        >
          {centerValue}
        </text>
        <text
          x="50%"
          y="60%"
          textAnchor="middle"
          className="fill-current text-forest-soft"
          style={{ fontSize: 10, letterSpacing: "0.08em" }}
        >
          {centerLabel}
        </text>
      </svg>
      <ul className="min-w-[10rem] flex-1 space-y-2.5">
        {data.map((d) => (
          <li key={d.label} className="flex items-center gap-2.5 text-sm">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ backgroundColor: d.color }}
              aria-hidden="true"
            />
            <span className="flex-1 text-forest">{d.label}</span>
            <span className="font-mono text-xs tabular-nums text-forest-mid">{n(d.value)}</span>
            <span className="w-11 text-right font-mono text-xs tabular-nums text-forest-soft">
              {total ? ((d.value / total) * 100).toFixed(1) : 0}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Vertical column histogram, growing from the baseline on first view. */
export function Histogram({ data, height = 150, unit = "", tone = "#3A4E41" }) {
  const [ref, inView] = useInView({ threshold: 0.25 });
  const peak = Math.max(...data.map((d) => d.value));

  return (
    <div ref={ref}>
      {/* the columns must stretch to the full track height, otherwise the
          percentage heights on the bars resolve against a content-sized box */}
      <div className="flex items-stretch gap-1.5" style={{ height }}>
        {data.map((d, i) => (
          <div key={d.label} className="group flex h-full flex-1 flex-col items-center justify-end">
            <span className="mb-1 font-mono text-[0.65rem] tabular-nums text-forest-soft opacity-0 transition-all duration-200 group-hover:-translate-y-0.5 group-hover:opacity-100">
              {n(d.value)}
            </span>
            <div
              className="w-full rounded-t-sm transition-all duration-[900ms] ease-[cubic-bezier(0.16,1,0.3,1)] group-hover:brightness-110"
              style={{
                height: inView ? `${peak ? Math.max((d.value / peak) * 100, 1.5) : 0}%` : "0%",
                backgroundColor: d.color || tone,
                transitionDelay: `${i * 60}ms`,
              }}
              title={`${d.label}: ${n(d.value)}${unit}`}
            />
          </div>
        ))}
      </div>
      <div className="mt-2 flex gap-1.5 border-t border-sand-deep/60 pt-2">
        {data.map((d) => (
          <div
            key={d.label}
            className="flex-1 text-center font-mono text-[0.65rem] tabular-nums text-forest-soft"
          >
            {d.label}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Comparison of one metric across labelled systems, with counted-up values. */
export function MetricBars({ data, domainMin = 0, domainMax = 100, unit = "%" }) {
  const [ref, inView] = useInView({ threshold: 0.2 });
  const span = domainMax - domainMin;

  return (
    <ul ref={ref} className="space-y-4">
      {data.map((d, i) => (
        <li key={d.label}>
          <div className="flex items-baseline justify-between gap-4">
            <span className={`text-sm ${d.best ? "font-semibold text-forest-deep" : "text-forest"}`}>
              {d.label}
            </span>
            <span
              className={`shrink-0 font-mono text-sm tabular-nums ${
                d.best ? "font-semibold text-merlot-mid" : "text-forest-mid"
              }`}
            >
              <CountUp value={d.value} decimals={1} duration={1100} suffix={unit} />
            </span>
          </div>
          <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-sand/70">
            <div
              className={`h-full rounded-full transition-[width] duration-[900ms] ease-[cubic-bezier(0.16,1,0.3,1)] ${
                d.best ? "bg-merlot-mid" : "bg-forest-mid/70"
              }`}
              style={{
                width: inView ? `${((d.value - domainMin) / span) * 100}%` : "0%",
                transitionDelay: `${i * 70}ms`,
              }}
            />
          </div>
          {d.note && <p className="mt-1 text-xs text-forest-soft">{d.note}</p>}
        </li>
      ))}
    </ul>
  );
}

/**
 * Numbered vertical timeline whose steps light up one after another once the
 * list scrolls into view.
 */
export function Timeline({ steps, tone = "light" }) {
  const [ref, inView] = useInView({ threshold: 0.2 });
  const [active, setActive] = useState(-1);
  const dark = tone === "dark";

  const run = useCallback(() => {
    if (reduced()) {
      setActive(steps.length);
      return undefined;
    }
    let i = 0;
    setActive(0);
    const timer = setInterval(() => {
      i += 1;
      setActive(i);
      if (i >= steps.length) clearInterval(timer);
    }, 260);
    return () => clearInterval(timer);
  }, [steps.length]);

  useEffect(() => {
    if (!inView) return undefined;
    return run();
  }, [inView, run]);

  return (
    <ol ref={ref} className="relative">
      <div
        aria-hidden="true"
        className={`absolute left-[15px] top-3 w-px ${dark ? "bg-sand/20" : "bg-sand-deep"}`}
        style={{ bottom: "1rem" }}
      />
      <div
        aria-hidden="true"
        className="absolute left-[15px] top-3 w-px bg-merlot-mid transition-[height] duration-700 ease-out"
        style={{
          height: `${Math.min(active / steps.length, 1) * 100}%`,
        }}
      />

      {steps.map((s, i) => {
        const on = i < active;
        return (
          <li
            key={s.step}
            className="relative flex gap-4 pb-5 last:pb-0"
            style={{
              opacity: on ? 1 : 0.35,
              transform: on ? "translateY(0)" : "translateY(6px)",
              transition: "opacity 500ms ease-out, transform 500ms ease-out",
            }}
          >
            <span
              className={`relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-[0.72rem] font-medium transition-all duration-500 ${
                on
                  ? "border-merlot-mid bg-merlot text-cream"
                  : dark
                  ? "border-sand/25 bg-forest text-sand/50"
                  : "border-sand-deep bg-ivory text-forest-soft"
              }`}
            >
              {i + 1}
            </span>
            <div className="pt-1">
              <p
                className={`text-sm font-medium ${dark ? "text-cream" : "text-forest-deep"}`}
              >
                {s.step}
              </p>
              <p className={`text-xs leading-relaxed ${dark ? "text-sand/60" : "text-forest-soft"}`}>
                {s.note}
              </p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
