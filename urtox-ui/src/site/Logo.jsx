import React from "react";

const SRC = `${process.env.PUBLIC_URL || ""}/mark.png`;
const ALPHA = `${process.env.PUBLIC_URL || ""}/mark-alpha.png`;

/**
 * The URTOX mark.
 *
 * By default this renders the artwork in its own green. Pass `tinted` to have
 * it take the surrounding `currentColor` instead, which is what the oversized
 * hero watermark needs so it can sit in sand rather than green.
 */
export default function Logo({ size = 28, className = "", title, tinted = false, style }) {
  if (tinted) {
    const mask = {
      WebkitMaskImage: `url(${ALPHA})`,
      maskImage: `url(${ALPHA})`,
      WebkitMaskRepeat: "no-repeat",
      maskRepeat: "no-repeat",
      WebkitMaskSize: "contain",
      maskSize: "contain",
      WebkitMaskPosition: "center",
      maskPosition: "center",
      backgroundColor: "currentColor",
      width: size,
      height: size,
      ...style,
    };
    return <span aria-hidden="true" className={className} style={mask} />;
  }

  return (
    <img
      src={SRC}
      width={size}
      height={size}
      alt={title || ""}
      aria-hidden={title ? undefined : true}
      className={className}
      style={{ objectFit: "contain", ...style }}
      draggable="false"
    />
  );
}
