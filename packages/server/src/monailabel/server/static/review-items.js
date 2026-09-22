// Reviews belong to a complete source or to its independently submitted regions/ranges.
export function reviewItems(state) {
  return [...state.assets, ...(state.videos || [])].flatMap((source) => {
    const units = (state.reviewUnits || []).filter(
      (u) => u.asset_id === source.id,
    );
    return units.length
      ? units
          .filter((u) => u.annotation_id)
          .map((unit) => ({
            ...source,
            ...unit,
            unit_id: unit.id,
            source_id: source.id,
            source_name: source.name,
            name: `${source.name} · ${unit.name}`,
          }))
      : source.annotation_id
        ? [source]
        : [];
  });
}

export function reviewStatus(state, source, latestDecision) {
  const units = source.unit_id
    ? []
    : (state.reviewUnits || []).filter((u) => u.asset_id === source.id);
  if (!units.length)
    return source.annotation_id
      ? latestDecision(source)?.verdict || "pending"
      : "unannotated";
  const verdicts = units.map(
    (unit) => latestDecision(unit)?.verdict || "pending",
  );
  return verdicts.includes("changes_requested")
    ? "changes_requested"
    : verdicts.includes("pending")
      ? "pending"
      : "accepted";
}

export function reviewSummary(state, source, latestDecision) {
  const units = (state.reviewUnits || []).filter(
    (u) => u.asset_id === source.id,
  );
  const names = {
    accepted: "accepted",
    pending: "pending",
    changes_requested: "need changes",
  };
  return Object.entries(names)
    .map(([verdict, name]) => {
      const count = units.filter(
        (u) => (latestDecision(u)?.verdict || "pending") === verdict,
      ).length;
      return count ? `${count} ${name}` : "";
    })
    .filter(Boolean)
    .join(" · ");
}
