using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Builds a map from a seed and a biome. Deterministic: the same seed and biome
    /// always give the same map, on any machine, because <see cref="Noise"/> is
    /// implemented rather than taken from the engine.
    ///
    /// The passes run in this order, and the order matters:
    ///   1. Elevation — one fractal field, the thing every later pass reads.
    ///   2. Water — everything under the waterline, which the biome sets.
    ///   3. Ground cover — bare soil, grass or forest, from a second noise field
    ///      against the biome's two thresholds.
    ///   4. Rock — high ground becomes stone, and the highest becomes impassable.
    ///   5. Camp site — the flattest, driest patch, levelled and cleared.
    ///   6. Boulders — scattered on open ground, never in the camp.
    ///
    /// The camp is levelled and kept clear of water, cliff and rock so it can be
    /// built on, but it is *not* stamped bare: it takes the same ground cover as the
    /// rest of the map. The HTML build had that wrong for a long time and the result
    /// was a permanent dirt scar that regrew at about four tiles a year.
    /// </summary>
    public static class WorldGenerator
    {
        /// <summary>Elevation above which ground is bare stone.</summary>
        public const float RockLine = 0.70f;

        /// <summary>Elevation above which the rock is a cliff face and cannot be crossed.</summary>
        public const float CliffLine = 0.82f;

        /// <summary>How many cells across the camp clearing is.</summary>
        public const int CampRadius = 6;

        public static WorldGenResult Generate(int width, int height, int seed, BiomeKind biomeKind)
        {
            var map = new Map(width, height);
            BiomeDef biome = BiomeDef.Of(biomeKind);
            var noise = new Noise(seed);

            float[] elevation = BuildElevation(map, noise);
            float waterLine = WaterLineFor(elevation, biome.Water);

            PaintTerrain(map, noise, biome, elevation, waterLine);

            Vector2Int campCentre = ChooseCampSite(map, elevation, waterLine);
            ClearCamp(map, noise, biome, elevation, campCentre);

            ScatterBoulders(map, noise, biome, campCentre);

            map.Links.RecomputeAll();

            return new WorldGenResult(map, biome, campCentre, elevation, waterLine);
        }

        private static float[] BuildElevation(Map map, Noise noise)
        {
            var elevation = new float[map.CellCount];

            // frequency is fixed per map rather than per tile, so a bigger map gets
            // bigger landforms instead of more of them. About five hills across the
            // longer side: enough to give a map a shape, few enough that a camp-sized
            // patch of flat ground exists somewhere on it.
            float f = 5f / Mathf.Max(map.Width, map.Height);

            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    float e = noise.Fractal(x * f, y * f, 5, 0.5f);

                    // a ridge at half the frequency, so maps have a spine running
                    // through them rather than only a field of blobs
                    float ridge = noise.Ridged(x * f * 0.5f, y * f * 0.5f, 3, 977);
                    e = Mathf.Lerp(e, ridge, 0.28f);

                    elevation[map.CellToIndex(x, y)] = Mathf.Clamp01(e);
                }
            }

            return elevation;
        }

        /// <summary>
        /// The elevation under which a given share of the map lies. Taking the quantile
        /// rather than a fixed height is what makes the biome's water share mean what it
        /// says on every seed, instead of some seeds coming out as swamp and others dry.
        /// </summary>
        private static float WaterLineFor(float[] elevation, float share)
        {
            if (share <= 0f)
                return -1f;

            var sorted = new float[elevation.Length];
            System.Array.Copy(elevation, sorted, elevation.Length);
            System.Array.Sort(sorted);

            int index = Mathf.Clamp(Mathf.FloorToInt(share * sorted.Length), 0, sorted.Length - 1);
            return sorted[index];
        }

        private static void PaintTerrain(Map map, Noise noise, BiomeDef biome, float[] elevation, float waterLine)
        {
            // cover varies faster than elevation: patches of scrub and wood within a
            // single hillside, not one biome per hill
            float coverScale = 9f / Mathf.Max(map.Width, map.Height);

            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    var cell = new Vector2Int(x, y);
                    float e = elevation[map.CellToIndex(x, y)];

                    if (e <= waterLine)
                    {
                        map.Terrain.Set(cell, TerrainKind.Water);
                        continue;
                    }

                    if (e >= CliffLine)
                    {
                        map.Terrain.Set(cell, TerrainKind.Rock);
                        map.Edifices.Set(cell, EdificeKind.NaturalRock);
                        continue;
                    }

                    if (e >= RockLine)
                    {
                        map.Terrain.Set(cell, TerrainKind.Rock);
                        continue;
                    }

                    float cover = noise.Fractal(x * coverScale, y * coverScale, 4, 0.5f, 313);

                    // sand rings a shore in the drier biomes, where there is no grass to hold it
                    bool shore = e < waterLine + 0.03f && biome.GroundCover > 0.5f;

                    map.Terrain.Set(cell,
                        shore ? TerrainKind.Sand
                        : cover > biome.TreeCover ? TerrainKind.Forest
                        : cover > biome.GroundCover ? TerrainKind.Grass
                        : TerrainKind.Soil);
                }
            }
        }

        /// <summary>
        /// Pick where the colony starts: the flattest patch that is dry and not stone.
        /// Flatness is what matters, because a camp on a slope cannot be levelled
        /// without a terrace, and the cost of looking is trivial next to arriving
        /// somewhere unbuildable.
        /// </summary>
        private static Vector2Int ChooseCampSite(Map map, float[] elevation, float waterLine)
        {
            Vector2Int best = new Vector2Int(map.Width / 2, map.Height / 2);
            float bestScore = float.MaxValue;

            int margin = CampRadius + 2;

            for (int y = margin; y < map.Height - margin; y++)
            {
                for (int x = margin; x < map.Width - margin; x++)
                {
                    float centre = elevation[map.CellToIndex(x, y)];

                    if (centre <= waterLine || centre >= RockLine)
                        continue;

                    // roughness: how much the ground moves over the camp's footprint
                    float roughness = 0f;
                    bool blocked = false;

                    for (int dy = -CampRadius; dy <= CampRadius && !blocked; dy++)
                    {
                        for (int dx = -CampRadius; dx <= CampRadius; dx++)
                        {
                            float e = elevation[map.CellToIndex(x + dx, y + dy)];

                            if (e >= CliffLine) { blocked = true; break; }

                            roughness += Mathf.Abs(e - centre);
                        }
                    }

                    if (blocked)
                        continue;

                    // a little height above the waterline is worth having: it keeps the
                    // camp out of the wet without pushing it up into the rock
                    float dryness = Mathf.Max(0f, 0.12f - (centre - waterLine)) * 40f;
                    float score = roughness + dryness;

                    if (score < bestScore)
                    {
                        bestScore = score;
                        best = new Vector2Int(x, y);
                    }
                }
            }

            return best;
        }

        private static void ClearCamp(Map map, Noise noise, BiomeDef biome, float[] elevation, Vector2Int centre)
        {
            float coverScale = 9f / Mathf.Max(map.Width, map.Height);

            for (int dy = -CampRadius; dy <= CampRadius; dy++)
            {
                for (int dx = -CampRadius; dx <= CampRadius; dx++)
                {
                    var cell = new Vector2Int(centre.x + dx, centre.y + dy);
                    if (!map.InBounds(cell))
                        continue;

                    // level the ground and take out anything that would block building
                    map.Edifices.Clear(cell);

                    // the camp takes the same cover as anywhere else of its fertility.
                    // Only forest is held back, because trees would block building —
                    // stamping the whole footprint bare is what left a dirt scar.
                    float cover = noise.Fractal(cell.x * coverScale, cell.y * coverScale, 4, 0.5f, 313);

                    map.Terrain.Set(cell, cover > biome.GroundCover ? TerrainKind.Grass : TerrainKind.Soil);
                }
            }
        }

        private static void ScatterBoulders(Map map, Noise noise, BiomeDef biome, Vector2Int campCentre)
        {
            if (biome.Rockiness <= 0f)
                return;

            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    var cell = new Vector2Int(x, y);

                    // never inside the camp clearing, or the colony starts walled in
                    if (Mathf.Abs(x - campCentre.x) <= CampRadius && Mathf.Abs(y - campCentre.y) <= CampRadius)
                        continue;

                    if (map.Edifices.IsStructure(cell))
                        continue;

                    TerrainKind t = map.Terrain.At(cell);
                    if (t == TerrainKind.Water || t == TerrainKind.Rock)
                        continue;

                    if (noise.Hash(x, y, 5501) < biome.Rockiness * 0.25f)
                        map.Edifices.Set(cell, EdificeKind.NaturalRock);
                }
            }
        }
    }

    public readonly struct WorldGenResult
    {
        public readonly Map Map;
        public readonly BiomeDef Biome;
        public readonly Vector2Int CampCentre;
        public readonly float[] Elevation;
        public readonly float WaterLine;

        public WorldGenResult(Map map, BiomeDef biome, Vector2Int campCentre, float[] elevation, float waterLine)
        {
            Map = map;
            Biome = biome;
            CampCentre = campCentre;
            Elevation = elevation;
            WaterLine = waterLine;
        }

        public float ElevationAt(Vector2Int cell) => Elevation[Map.CellToIndex(cell)];
    }
}
