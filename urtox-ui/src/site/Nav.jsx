import React, { useEffect, useState } from "react";
import { Menu, X, ArrowUpRight } from "lucide-react";
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

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState("");

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Highlight whichever section currently occupies the upper part of the viewport.
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return undefined;
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
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 transition-colors duration-300 ${
        scrolled ? "border-b border-sand-deep/60 bg-ivory/92 backdrop-blur-md" : "bg-transparent"
      }`}
    >
      <nav
        aria-label="Primary"
        className="mx-auto flex max-w-6xl items-center gap-4 px-5 py-3.5 sm:px-8"
      >
        <a href="#top" className="group flex items-center gap-2.5">
          <Logo
            size={28}
            title="URTOX"
            className="shrink-0 transition-transform duration-300 group-hover:scale-105"
          />
          <span className="font-serif text-lg font-medium leading-none tracking-tight text-forest-deep">
            URTOX
          </span>
          <span className="hidden text-xs leading-none text-forest-soft sm:inline">
            Urdu Toxic Span Dataset
          </span>
        </a>

        <ul className="ml-auto hidden items-center gap-1 lg:flex">
          {ITEMS.map((item) => (
            <li key={item.href}>
              <a
                href={item.href}
                className={`rounded px-2.5 py-1.5 text-[0.82rem] transition-colors ${
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

        <a
          href={LINKS.dataset}
          target="_blank"
          rel="noreferrer"
          className="ml-auto hidden items-center gap-1.5 rounded-md bg-forest px-3.5 py-2 text-[0.82rem] font-medium text-cream transition-colors hover:bg-forest-deep lg:ml-0 lg:inline-flex"
        >
          Get the data
          <ArrowUpRight size={14} aria-hidden="true" />
        </a>

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={open ? "Close menu" : "Open menu"}
          className="ml-auto rounded-md border border-sand-deep p-2 text-forest lg:hidden"
        >
          {open ? <X size={18} /> : <Menu size={18} />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-sand-deep/60 bg-ivory lg:hidden">
          <ul className="mx-auto grid max-w-6xl grid-cols-2 gap-1 px-5 py-3 sm:px-8">
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
            <li className="col-span-2 mt-1">
              <a
                href={LINKS.dataset}
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-center gap-1.5 rounded-md bg-forest px-3 py-2.5 text-sm font-medium text-cream"
              >
                Get the data
                <ArrowUpRight size={14} aria-hidden="true" />
              </a>
            </li>
          </ul>
        </div>
      )}
    </header>
  );
}
