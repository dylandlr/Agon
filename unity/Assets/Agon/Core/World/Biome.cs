using UnityEngine;

namespace Agon.Core
{
    public enum BiomeKind : byte
    {
        Temperate = 0,
        Desert = 1,
        Tundra = 2,
        Savanna = 3,
        Jungle = 4,
        Taiga = 5
    }

    /// <summary>
    /// What a biome does to worldgen and to the simulation. The fields are the ones
    /// generation actually reads, so a new biome is a row here rather than a branch
    /// somewhere in the generator.
    ///
    /// <see cref="GroundCover"/> and <see cref="TreeCover"/> are thresholds on the same
    /// coverage noise: below ground cover is bare soil, above it is grass, and above
    /// tree cover it is forest. A desert simply has both set high enough that almost
    /// nothing clears them, which is why its camps come out bare without any
    /// desert-specific code.
    /// </summary>
    public readonly struct BiomeDef
    {
        public readonly BiomeKind Kind;
        public readonly string Label;

        /// <summary>Coverage above which bare soil becomes grass. Higher means barer.</summary>
        public readonly float GroundCover;

        /// <summary>Coverage above which grass becomes forest. Higher means fewer trees.</summary>
        public readonly float TreeCover;

        /// <summary>Mean temperature in degrees, before the season swing.</summary>
        public readonly float Temperature;

        /// <summary>How far temperature swings between midwinter and midsummer.</summary>
        public readonly float SeasonSwing;

        /// <summary>Share of the map below the waterline, as lakes and rivers.</summary>
        public readonly float Water;

        /// <summary>Scatter of loose boulders, 0..1.</summary>
        public readonly float Rockiness;

        public BiomeDef(BiomeKind kind, string label, float groundCover, float treeCover,
                        float temperature, float seasonSwing, float water, float rockiness)
        {
            Kind = kind;
            Label = label;
            GroundCover = groundCover;
            TreeCover = treeCover;
            Temperature = temperature;
            SeasonSwing = seasonSwing;
            Water = water;
            Rockiness = rockiness;
        }

        private static readonly BiomeDef[] All =
        {
            //                                         ground  tree   temp  swing  water  rock
            new BiomeDef(BiomeKind.Temperate, "Temperate", 0.38f, 0.62f,  11f,  14f, 0.10f, 0.05f),
            new BiomeDef(BiomeKind.Desert,    "Desert",    0.86f, 0.97f,  28f,  16f, 0.01f, 0.12f),
            new BiomeDef(BiomeKind.Tundra,    "Tundra",    0.74f, 0.93f,  -8f,  18f, 0.05f, 0.10f),
            new BiomeDef(BiomeKind.Savanna,   "Savanna",   0.60f, 0.88f,  24f,   9f, 0.03f, 0.07f),
            new BiomeDef(BiomeKind.Jungle,    "Jungle",    0.20f, 0.40f,  26f,   5f, 0.14f, 0.03f),
            new BiomeDef(BiomeKind.Taiga,     "Taiga",     0.34f, 0.52f,   1f,  20f, 0.11f, 0.08f)
        };

        public static BiomeDef Of(BiomeKind kind) => All[(int)kind];

        public static int Count => All.Length;

        /// <summary>
        /// Temperature at a tick, swinging over the year. Season 0 is the coldest part
        /// of the year, so the curve starts at its trough rather than its mean.
        /// </summary>
        public float TemperatureAt(int ticksGame)
        {
            float yearFraction = (ticksGame % TickManager.TicksPerYear) / (float)TickManager.TicksPerYear;
            return Temperature - Mathf.Cos(yearFraction * Mathf.PI * 2f) * SeasonSwing * 0.5f;
        }
    }
}
