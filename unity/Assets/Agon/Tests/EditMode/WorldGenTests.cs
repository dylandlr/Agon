using NUnit.Framework;
using UnityEngine;
using Agon.Core;

namespace Agon.Tests
{
    /// <summary>
    /// What worldgen guarantees. Every expectation here was first checked against an
    /// independent re-implementation over 48 generated worlds, so these state the
    /// intended properties rather than describing whatever came out.
    /// </summary>
    public class WorldGenTests
    {
        private const int W = 80, H = 50;

        private static float Share(Map map, TerrainKind kind)
        {
            int n = 0;
            for (int y = 0; y < map.Height; y++)
                for (int x = 0; x < map.Width; x++)
                    if (map.Terrain.At(new Vector2Int(x, y)) == kind) n++;

            return n / (float)map.CellCount;
        }

        [Test]
        public void SameSeed_GivesTheSameWorld_AndADifferentSeedDoesNot()
        {
            WorldGenResult a = WorldGenerator.Generate(W, H, 3, BiomeKind.Temperate);
            WorldGenResult b = WorldGenerator.Generate(W, H, 3, BiomeKind.Temperate);
            WorldGenResult c = WorldGenerator.Generate(W, H, 4, BiomeKind.Temperate);

            Assert.AreEqual(a.CampCentre, b.CampCentre, "One seed, one camp site.");

            bool identical = true, differs = false;
            for (int i = 0; i < a.Map.CellCount; i++)
            {
                Vector2Int cell = a.Map.IndexToCell(i);
                if (a.Map.Terrain.At(cell) != b.Map.Terrain.At(cell)) identical = false;
                if (a.Map.Terrain.At(cell) != c.Map.Terrain.At(cell)) differs = true;
            }

            Assert.IsTrue(identical, "The same seed must reproduce the map exactly.");
            Assert.IsTrue(differs, "A different seed must produce a different map.");
        }

        [Test]
        public void WaterShare_FollowsTheBiome()
        {
            // the waterline is taken as a quantile of the elevation field rather than a
            // fixed height, which is what makes this true on every seed instead of some
            // coming out as swamp and others bone dry
            void Check(BiomeKind kind, float expected)
            {
                WorldGenResult w = WorldGenerator.Generate(W, H, 7, kind);
                Assert.AreEqual(expected, Share(w.Map, TerrainKind.Water), 0.02f,
                    kind + " should be about " + expected + " water.");
            }

            Check(BiomeKind.Temperate, 0.10f);
            Check(BiomeKind.Jungle, 0.14f);
            Check(BiomeKind.Desert, 0.01f);
            Check(BiomeKind.Savanna, 0.03f);
        }

        [Test]
        public void BiomesLookDifferentFromEachOther()
        {
            WorldGenResult desert = WorldGenerator.Generate(W, H, 11, BiomeKind.Desert);
            WorldGenResult jungle = WorldGenerator.Generate(W, H, 11, BiomeKind.Jungle);

            Assert.Greater(Share(desert.Map, TerrainKind.Soil), Share(jungle.Map, TerrainKind.Soil),
                "Desert ground should be barer.");

            Assert.Greater(Share(jungle.Map, TerrainKind.Forest), Share(desert.Map, TerrainKind.Forest) + 0.2f,
                "Jungle should be far more wooded.");
        }

        [Test]
        public void TheCampSite_IsAlwaysBuildable()
        {
            for (int b = 0; b < BiomeDef.Count; b++)
            {
                for (int seed = 1; seed <= 8; seed++)
                {
                    WorldGenResult w = WorldGenerator.Generate(W, H, seed, (BiomeKind)b);
                    Vector2Int c = w.CampCentre;

                    for (int dy = -WorldGenerator.CampRadius; dy <= WorldGenerator.CampRadius; dy++)
                    {
                        for (int dx = -WorldGenerator.CampRadius; dx <= WorldGenerator.CampRadius; dx++)
                        {
                            var cell = new Vector2Int(c.x + dx, c.y + dy);
                            TerrainKind t = w.Map.Terrain.At(cell);
                            string where = (BiomeKind)b + " seed " + seed + " at " + cell;

                            Assert.AreNotEqual(TerrainKind.Water, t, "Camp under water: " + where);
                            Assert.AreNotEqual(TerrainKind.Rock, t, "Camp on bare rock: " + where);
                            Assert.AreNotEqual(TerrainKind.Forest, t, "Trees would block building: " + where);
                            Assert.IsFalse(w.Map.Edifices.IsStructure(cell), "Boulder in the camp: " + where);
                            Assert.IsTrue(w.Map.Terrain.Walkable(cell), "Camp cell not walkable: " + where);
                        }
                    }
                }
            }
        }

        [Test]
        public void AGreenBiome_DoesNotGetABareDirtCamp()
        {
            // the regression this prevents: the HTML build ran the terrain decision only
            // outside the camp, so every camp tile fell through to bare soil and stayed
            // a dirt scar that regrew at about four tiles a year
            foreach (BiomeKind kind in new[] { BiomeKind.Temperate, BiomeKind.Jungle, BiomeKind.Taiga })
            {
                for (int seed = 1; seed <= 8; seed++)
                {
                    WorldGenResult w = WorldGenerator.Generate(W, H, seed, kind);
                    Vector2Int c = w.CampCentre;

                    int grass = 0;
                    for (int dy = -WorldGenerator.CampRadius; dy <= WorldGenerator.CampRadius; dy++)
                        for (int dx = -WorldGenerator.CampRadius; dx <= WorldGenerator.CampRadius; dx++)
                            if (w.Map.Terrain.At(new Vector2Int(c.x + dx, c.y + dy)) == TerrainKind.Grass) grass++;

                    Assert.Greater(grass, 0, kind + " seed " + seed + ": a green biome's camp should have grass in it.");
                }
            }
        }

        [Test]
        public void TheCampSite_IsOnFlatGroundAndInland()
        {
            for (int seed = 1; seed <= 8; seed++)
            {
                WorldGenResult w = WorldGenerator.Generate(W, H, seed, BiomeKind.Temperate);
                Vector2Int c = w.CampCentre;

                Assert.GreaterOrEqual(c.x, WorldGenerator.CampRadius + 2, "Camp too close to the west edge.");
                Assert.GreaterOrEqual(c.y, WorldGenerator.CampRadius + 2, "Camp too close to the south edge.");
                Assert.Less(c.x, W - WorldGenerator.CampRadius - 2, "Camp too close to the east edge.");
                Assert.Less(c.y, H - WorldGenerator.CampRadius - 2, "Camp too close to the north edge.");

                float centre = w.ElevationAt(c);
                float relief = 0f;

                for (int dy = -WorldGenerator.CampRadius; dy <= WorldGenerator.CampRadius; dy++)
                    for (int dx = -WorldGenerator.CampRadius; dx <= WorldGenerator.CampRadius; dx++)
                        relief = Mathf.Max(relief, Mathf.Abs(w.ElevationAt(new Vector2Int(c.x + dx, c.y + dy)) - centre));

                Assert.Less(relief, 0.25f, "Seed " + seed + ": the camp should sit on flat ground.");
            }
        }

        [Test]
        public void Elevation_HasRealRelief_AndIsNotDegenerate()
        {
            WorldGenResult w = WorldGenerator.Generate(W, H, 3, BiomeKind.Temperate);

            float min = float.MaxValue, max = float.MinValue, sum = 0f;
            foreach (float e in w.Elevation)
            {
                min = Mathf.Min(min, e);
                max = Mathf.Max(max, e);
                sum += e;
            }

            Assert.Greater(max - min, 0.35f, "A map with no relief has nothing to route around.");

            float mean = sum / w.Elevation.Length;
            Assert.That(mean, Is.GreaterThan(0.2f).And.LessThan(0.8f),
                "Elevation should fill its range rather than pile up at one end.");
        }

        [Test]
        public void ABiggerMap_GetsBiggerLandforms_NotProportionallyMoreOfThem()
        {
            int Peaks(WorldGenResult w)
            {
                int n = 0;
                for (int y = 1; y < w.Map.Height - 1; y++)
                {
                    for (int x = 1; x < w.Map.Width - 1; x++)
                    {
                        float e = w.ElevationAt(new Vector2Int(x, y));
                        bool top = true;

                        for (int d = 0; d < 4; d++)
                            if (w.ElevationAt(new Vector2Int(x, y) + Map.CardinalDirections[d]) >= e) top = false;

                        if (top) n++;
                    }
                }

                return n;
            }

            int small = Peaks(WorldGenerator.Generate(60, 40, 5, BiomeKind.Temperate));
            int big = Peaks(WorldGenerator.Generate(120, 80, 5, BiomeKind.Temperate));

            Assert.Less(big, small * 4,
                "Frequency is per map, not per tile, so a map four times the area must not get four times the hills.");
        }

        [Test]
        public void CliffsAreImpassable_ButPlainRockIsNot()
        {
            WorldGenResult w = WorldGenerator.Generate(W, H, 3, BiomeKind.Tundra);

            int cliffs = 0;
            for (int i = 0; i < w.Map.CellCount; i++)
            {
                Vector2Int cell = w.Map.IndexToCell(i);
                if (w.ElevationAt(cell) < WorldGenerator.CliffLine)
                    continue;

                cliffs++;
                Assert.IsFalse(w.Map.Terrain.Walkable(cell), "A cliff face has to block movement.");
            }

            Assert.Greater(cliffs, 0, "The fixture should contain some cliff, or this proves nothing.");
        }

        [Test]
        public void Temperature_SwingsAcrossTheYear()
        {
            BiomeDef tundra = BiomeDef.Of(BiomeKind.Tundra);

            float midwinter = tundra.TemperatureAt(0);
            float midsummer = tundra.TemperatureAt(TickManager.TicksPerYear / 2);

            Assert.Less(midwinter, midsummer, "Season 0 is the cold end of the year.");
            Assert.AreEqual(tundra.SeasonSwing, midsummer - midwinter, 0.5f,
                "The swing between the extremes should be the biome's stated swing.");

            BiomeDef jungle = BiomeDef.Of(BiomeKind.Jungle);
            Assert.Less(jungle.TemperatureAt(TickManager.TicksPerYear / 2) - jungle.TemperatureAt(0),
                        tundra.SeasonSwing, "A jungle's year should be far more even than a tundra's.");
        }
    }

    public class NoiseTests
    {
        [Test]
        public void Noise_IsDeterministic_AndSeedDependent()
        {
            var a = new Noise(7);
            var b = new Noise(7);
            var c = new Noise(8);

            Assert.AreEqual(a.Value(3.3f, 1.7f), b.Value(3.3f, 1.7f), 1e-6f);
            Assert.AreNotEqual(a.Value(3.3f, 1.7f), c.Value(3.3f, 1.7f), 1e-6f);
        }

        [Test]
        public void Noise_StaysInRange_AndIsNotConstant()
        {
            var noise = new Noise(42);
            float min = float.MaxValue, max = float.MinValue;

            for (int y = 0; y < 40; y++)
            {
                for (int x = 0; x < 40; x++)
                {
                    float v = noise.Fractal(x * 0.17f, y * 0.17f, 4, 0.5f);
                    Assert.That(v, Is.GreaterThanOrEqualTo(0f).And.LessThanOrEqualTo(1f));
                    min = Mathf.Min(min, v);
                    max = Mathf.Max(max, v);
                }
            }

            Assert.Greater(max - min, 0.2f, "Flat noise would make every map identical.");
        }

        [Test]
        public void ASeedOfZero_StillProducesVaryingNoise()
        {
            // 0 would collapse the hash to a constant, so the constructor substitutes one
            var noise = new Noise(0);

            Assert.AreNotEqual(noise.Hash(1, 1), noise.Hash(2, 2),
                "A zero seed must not degenerate to a constant field.");
        }
    }
}
