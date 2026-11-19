using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// What a wall is built from. Ordered by the tech that unlocks it, so a later
    /// material is also the stronger one and <c>&gt;</c> compares as "better".
    /// </summary>
    public enum WallMaterial : byte
    {
        Wattle = 0,
        Timber = 1,
        Stone  = 2
    }

    /// <summary>
    /// The properties of each wall material. Insulation and hit points carry the
    /// values the HTML build uses (WALLMATS), so the two implementations agree on
    /// the simulation's numbers.
    /// </summary>
    public readonly struct WallMaterialDef
    {
        public readonly WallMaterial Material;
        public readonly string Label;

        /// <summary>Degrees of warmth a closed room of this material holds.</summary>
        public readonly float Insulation;

        /// <summary>Relative toughness against damage: wattle 1, timber 2, stone 4.</summary>
        public readonly float HitPointsFactor;

        /// <summary>Body colour, used for the flat corner patch and as a tint fallback.</summary>
        public readonly Color Color;

        public WallMaterialDef(WallMaterial material, string label, float insulation, float hitPointsFactor, Color color)
        {
            Material = material;
            Label = label;
            Insulation = insulation;
            HitPointsFactor = hitPointsFactor;
            Color = color;
        }

        private static readonly WallMaterialDef[] All =
        {
            new WallMaterialDef(WallMaterial.Wattle, "Wattle", 8f,  1f, new Color32(0xB8, 0xA2, 0x7C, 0xFF)),
            new WallMaterialDef(WallMaterial.Timber, "Timber", 10f, 2f, new Color32(0x6F, 0x52, 0x38, 0xFF)),
            new WallMaterialDef(WallMaterial.Stone,  "Stone",  12f, 4f, new Color32(0x5F, 0x5C, 0x58, 0xFF))
        };

        public static WallMaterialDef Of(WallMaterial material) => All[(int)material];

        public static int Count => All.Length;
    }
}
