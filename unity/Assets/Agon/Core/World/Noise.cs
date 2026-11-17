using UnityEngine;

namespace Agon.Core
{
    /// <summary>
    /// Value noise with fractal octaves, seeded and deterministic.
    ///
    /// Unity's <see cref="Mathf.PerlinNoise"/> is deliberately not used: it is not
    /// guaranteed identical across platforms or Unity versions, so a world generated
    /// from a seed could differ between a player's machine and a test run. A world
    /// that cannot be reproduced from its seed cannot be debugged from a bug report,
    /// so the noise is implemented here where its behaviour is fixed.
    /// </summary>
    public struct Noise
    {
        private readonly uint _seed;

        public Noise(int seed)
        {
            // never let the seed be 0, which would collapse the hash to a constant
            _seed = (uint)seed == 0u ? 0x9E3779B9u : (uint)seed;
        }

        /// <summary>Hash of a lattice point to 0..1. Pure: the same inputs always give the same value.</summary>
        public float Hash(int x, int y, int channel = 0)
        {
            unchecked
            {
                uint h = _seed;
                h ^= (uint)x * 0x8DA6B343u;
                h ^= (uint)y * 0xD8163841u;
                h ^= (uint)channel * 0xCB1AB31Fu;

                // finalizer, so neighbouring lattice points do not correlate
                h ^= h >> 16;
                h *= 0x7FEB352Du;
                h ^= h >> 15;
                h *= 0x846CA68Bu;
                h ^= h >> 16;

                return (h & 0xFFFFFFu) / (float)0x1000000;
            }
        }

        /// <summary>Smoothed value noise at a point, with 1.0 being one lattice cell.</summary>
        public float Value(float x, float y, int channel = 0)
        {
            int x0 = Mathf.FloorToInt(x), y0 = Mathf.FloorToInt(y);
            float fx = x - x0, fy = y - y0;

            // smoothstep the interpolant, or the lattice shows as a grid of creases
            float sx = fx * fx * (3f - 2f * fx);
            float sy = fy * fy * (3f - 2f * fy);

            float a = Mathf.Lerp(Hash(x0, y0, channel), Hash(x0 + 1, y0, channel), sx);
            float b = Mathf.Lerp(Hash(x0, y0 + 1, channel), Hash(x0 + 1, y0 + 1, channel), sx);

            return Mathf.Lerp(a, b, sy);
        }

        /// <summary>
        /// Fractal noise: several octaves of <see cref="Value"/>, each twice the
        /// frequency and half the amplitude of the last. Normalized to 0..1.
        /// </summary>
        public float Fractal(float x, float y, int octaves = 4, float persistence = 0.5f, int channel = 0)
        {
            float sum = 0f, amplitude = 1f, frequency = 1f, total = 0f;

            for (int i = 0; i < octaves; i++)
            {
                sum += Value(x * frequency, y * frequency, channel + i * 31) * amplitude;
                total += amplitude;
                amplitude *= persistence;
                frequency *= 2f;
            }

            return total > 0f ? sum / total : 0f;
        }

        /// <summary>Ridged noise, for ranges and watercourses: folded so the midline becomes a crest.</summary>
        public float Ridged(float x, float y, int octaves = 4, int channel = 0)
        {
            float v = Fractal(x, y, octaves, 0.5f, channel);
            return 1f - Mathf.Abs(v * 2f - 1f);
        }
    }
}
