import React, { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";

import Nav from "./site/Nav";
import Hero from "./site/Hero";
import Claims from "./site/Claims";
import Motivation from "./site/Motivation";
import DatasetOverview from "./site/DatasetOverview";
import Annotation from "./site/Annotation";
import Explorer from "./site/Explorer";
import Challenges from "./site/Challenges";
import Architecture from "./site/Architecture";
import Results from "./site/Results";
import Multimodal from "./site/Multimodal";
import Pipeline from "./site/Pipeline";
import Publications from "./site/Publications";
import Access from "./site/Access";
import Citation from "./site/Citation";
import Reproducibility from "./site/Reproducibility";
import FutureWork from "./site/FutureWork";
import Footer from "./site/Footer";
import { FUTURE_WORK } from "./data/research";

import MutexWithXAI from "./MutexWithXAI.jsx";

/**
 * Two views, switched on the hash so no router dependency is needed:
 *   #/demo  the live inference demo backed by the deployed API
 *   default the research hub
 */
export default function App() {
  const [route, setRoute] = useState(
    typeof window !== "undefined" && window.location.hash.startsWith("#/demo") ? "demo" : "hub"
  );

  useEffect(() => {
    const onHash = () =>
      setRoute(window.location.hash.startsWith("#/demo") ? "demo" : "hub");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (route === "demo") {
    return (
      <>
        <div className="border-b border-sand-deep/60 bg-ivory">
          <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3 sm:px-8">
            <a
              href="#top"
              onClick={() => {
                window.location.hash = "";
              }}
              className="inline-flex items-center gap-2 text-sm font-medium text-forest hover:text-merlot-mid"
            >
              <ArrowLeft size={15} aria-hidden="true" />
              Back to the research hub
            </a>
            <span className="text-xs text-forest-soft">
              Live inference demo, requires the model API to be running
            </span>
          </div>
        </div>
        <MutexWithXAI />
      </>
    );
  }

  return (
    <>
      <a
        href="#motivation"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-forest focus:px-4 focus:py-2 focus:text-sm focus:text-cream"
      >
        Skip to content
      </a>

      <Nav />

      <main>
        <Hero />
        <Claims />
        <Motivation />
        <DatasetOverview />
        <Annotation />
        <Explorer />
        <Challenges />
        <Architecture />
        <Results />
        <Multimodal />
        <Pipeline />
        <Publications />
        <Access />
        <Citation />
        <Reproducibility />
        <FutureWork items={FUTURE_WORK} />
      </main>

      <Footer />
    </>
  );
}
