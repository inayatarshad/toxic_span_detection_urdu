import React, { useEffect, useState } from "react";

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
import Detector from "./site/Detector";
import Footer from "./site/Footer";
import { FUTURE_WORK } from "./data/research";

/**
 * Two equal surfaces sharing one shell, switched on the hash so no router
 * dependency is needed:
 *
 *   #/detector   the live toxic span detector, backed by the model API
 *   default      the dataset and research hub
 */
const isDetector = (hash) => hash.startsWith("#/detector") || hash.startsWith("#/demo");

export default function App() {
  const [route, setRoute] = useState(
    typeof window !== "undefined" && isDetector(window.location.hash) ? "detector" : "hub"
  );

  useEffect(() => {
    const onHash = () => {
      setRoute(isDetector(window.location.hash) ? "detector" : "hub");
      // switching surface should start at the top, not mid-scroll
      if (isDetector(window.location.hash)) window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-forest focus:px-4 focus:py-2 focus:text-sm focus:text-cream"
      >
        Skip to content
      </a>

      <Nav route={route} />

      <main id="main">
        {route === "detector" ? (
          <Detector />
        ) : (
          <>
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
          </>
        )}
      </main>

      <Footer />
    </>
  );
}
