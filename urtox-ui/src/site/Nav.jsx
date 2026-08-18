import React, { useEffect, useState } from "react";
import { Menu, X, ArrowUpRight, Zap, LibraryBig } from "lucide-react";
import { LINKS } from "../data/research";
import Logo from "./Logo";

const ITEMS = [
  { href: "#motivation", label: "Motivation" },
  { href: "#dataset", label: "Dataset" },
  { href: "#annotation", label: "Annotation" },
  { href: "#explore", label: "Explore" },
  { href: "#method", label: "Method" },
  { href: "#results", label: "Results" },
  { href: "#publications", label: "Publications" },
  { href: "#citation", label: "Cite" },
];

export default function Nav({ route = "hub" }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState("");
  const onHub = route === "hub";

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Highlight whichever section currently occupies the upper viewport.
  useEffect(() => {
    if (!onHub || typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActive(`#${visible.target.id}`);
      },
      { rootMargin: "-72px 0px -60% 0px" }
    );
    ITEMS.forEach(({ href }) => {
      const el = document.querySelector(href);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [onHub]);

  return (
    <header
      className={`sticky top-0 z-50 transition-colors duration-300 ${
        scrolled || !onHub
          ? "border-b border-sand-deep/60 bg-ivory/92 backdrop-blur-md"
          : "bg-transparent"
      }`}
    >
      <nav
        aria-label="Primary"
        className="mx-auto flex max-w-6xl items-center gap-4 px-5 py-3 sm:px-8"
      >
        <a href="#top" className="group flex shrink-0 items-center gap-2.5">
          <Logo
            size={28}
            title="URTOX"
            className="shrink-0 transition-transform duration-300 group-hover:scale-105"
          />
          <span className="font-serif text-lg font-medium leading-none tracking-tight text-forest-deep">
            URTOX
          </span>
        </a>

        {/* surface switch, so neither view is subordinate to the other */}
        <div
          role="tablist"
          aria-label="Choose a view"
          className="ml-1 hidden shrink-0 items-center gap-0.5 rounded-lg border border-sand-deep/70 bg-sand/30 p-0.5 sm:flex"
        >
          <a
            role="tab"
            aria-selected={onHub}
            href="#top"
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[0.78rem] font-medium transition-colors ${
              onHub ? "bg-forest text-cream" : "text-forest-mid hover:text-forest-deep"
            }`}
          >
            <LibraryBig size={13} aria-hidden="true" />
            Research hub
          </a>
          <a
            role="tab"
            aria-selected={!onHub}
            href="#/detector"
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[0.78rem] font-medium transition-colors ${
              !onHub ? "bg-merlot text-cream" : "text-forest-mid hover:text-forest-deep"
            }`}
          >
            <Zap size={13} aria-hidden="true" />
            Live detector
          </a>
        </div>

        {onHub && (
          <ul className="ml-auto hidden items-center gap-0.5 xl:flex">
            {ITEMS.map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  className={`rounded px-2 py-1.5 text-[0.78rem] transition-colors ${
                    active === item.href
                      ? "text-merlot-mid"
                      : "text-forest-mid hover:text-forest-deep"
                  }`}
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        )}

        <a
          href={LINKS.dataset}
          target="_blank"
          rel="noreferrer"
          className={`hidden shrink-0 items-center gap-1.5 rounded-md border border-sand-deep bg-white/70 px-3 py-2 text-[0.78rem] font-medium text-forest transition-colors hover:border-forest-soft lg:inline-flex ${
            onHub ? "" : "ml-auto"
          }`}
        >
          Get the data
          <ArrowUpRight size={13} aria-hidden="true" />
        </a>

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={open ? "Close menu" : "Open menu"}
          className="ml-auto rounded-md border border-sand-deep p-2 text-forest xl:hidden"
        >
          {open ? <X size={18} /> : <Menu size={18} />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-sand-deep/60 bg-ivory xl:hidden">
          <div className="mx-auto max-w-6xl px-5 py-3 sm:px-8">
            <div className="mb-2 grid grid-cols-2 gap-2 sm:hidden">
              <a
                href="#top"
                onClick={() => setOpen(false)}
                className={`flex items-center justify-center gap-1.5 rounded-md px-3 py-2.5 text-sm font-medium ${
                  onHub ? "bg-forest text-cream" : "border border-sand-deep text-forest"
                }`}
              >
                <LibraryBig size={14} aria-hidden="true" />
                Research hub
              </a>
              <a
                href="#/detector"
                onClick={() => setOpen(false)}
                className={`flex items-center justify-center gap-1.5 rounded-md px-3 py-2.5 text-sm font-medium ${
                  !onHub ? "bg-merlot text-cream" : "border border-sand-deep text-forest"
                }`}
              >
                <Zap size={14} aria-hidden="true" />
                Live detector
              </a>
            </div>

            {onHub && (
              <ul className="grid grid-cols-2 gap-1">
                {ITEMS.map((item) => (
                  <li key={item.href}>
                    <a
                      href={item.href}
                      onClick={() => setOpen(false)}
                      className="block rounded px-2 py-2.5 text-sm text-forest-mid hover:bg-sand/40"
                    >
                      {item.label}
                    </a>
                  </li>
                ))}
              </ul>
            )}

            <a
              href={LINKS.dataset}
              target="_blank"
              rel="noreferrer"
              className="mt-2 flex items-center justify-center gap-1.5 rounded-md border border-sand-deep px-3 py-2.5 text-sm font-medium text-forest"
            >
              Get the data
              <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          </div>
        </div>
      )}
    </header>
  );
}
