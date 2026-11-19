using System.Collections.Generic;
using UnityEngine;
using Agon.Core;
using Agon.Rendering;

namespace Agon.Bootstrap
{
    /// <summary>
    /// Brings a playable map up from nothing: drop this on an empty GameObject in an
    /// empty scene, press Play, and it builds the map, a camp to look at, the pawns
    /// and the camera. Nothing else in the scene is needed, and no art assets are —
    /// the wall atlas is drawn at runtime.
    ///
    /// This is scaffolding for the port, not the final game loop: worldgen, jobs and
    /// the rest of the HTML build's systems land alongside it.
    /// </summary>
    [AddComponentMenu("Agon/Agon Game")]
    public class AgonGame : MonoBehaviour
    {
        [Header("Map")]
        public int MapWidth = 80;
        public int MapHeight = 50;
        public int Seed = 3;
        public BiomeKind Biome = BiomeKind.Temperate;

        [Header("Simulation")]
        public TimeSpeed StartSpeed = TimeSpeed.Normal;
        public int PawnCount = 6;

        public Map Map { get; private set; }

        /// <summary>The generated world: elevation, biome and where the camp was sited.</summary>
        public WorldGenResult World => _world;

        private WorldGenResult _world;
        public TickManager Ticks { get; private set; }
        public IReadOnlyList<Pawn> Pawns => _pawns;

        private readonly List<Pawn> _pawns = new List<Pawn>();
        private WallRenderer _wallRenderer;

        private void Awake()
        {
            _world = WorldGenerator.Generate(MapWidth, MapHeight, Seed, Biome);
            Map = _world.Map;
            Ticks = new TickManager { CurTimeSpeed = StartSpeed };

            GenerateCamp();
            PlaceThings();
            SpawnPawns();

            Map.Links.RecomputeAll();

            _wallRenderer = gameObject.AddComponent<WallRenderer>();
            _wallRenderer.Initialize(Map);

            SetUpCamera();
        }

        private void Update()
        {
            Ticks.Update(Time.deltaTime);
        }

        /// <summary>One line per pawn: where it is, what it is doing, and how it is faring.</summary>
        public string DebugReport()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("tick ").Append(Ticks.TicksGame).Append("  day ").Append(Ticks.DayOfYear)
              .Append("  season ").Append(Ticks.Season).Append('\n');

            foreach (Pawn p in _pawns)
            {
                sb.Append(p.Id).Append(p.Alive ? "  " : " (dead) ")
                  .Append(p.Position).Append("  ")
                  .Append(p.Jobs.CurrentJob != null ? p.Jobs.CurrentJob.ToString() : "-")
                  .Append("  food ").Append(p.Needs.Food.CurLevel.ToString("0.00"))
                  .Append(" water ").Append(p.Needs.Water.CurLevel.ToString("0.00"))
                  .Append(" rest ").Append(p.Needs.Rest.CurLevel.ToString("0.00"))
                  .Append(" hp ").Append(p.Health.ToString("0.00"))
                  .Append("  done ").Append(p.Jobs.JobsDone).Append('\n');
            }

            return sb.ToString();
        }


        /// <summary>
        /// A camp laid out to exercise every wall case at once: two rooms sharing a
        /// wall, a doorway, and a 2x2 block of wall. The block is the one that proves
        /// the corner filler — without it there is a hole at the block's centre.
        /// </summary>
        private void GenerateCamp()
        {
            // build at the site worldgen levelled and cleared for us, not at the
            // map's centre, which may well be a lake or a cliff
            int ox = _world.CampCentre.x - 9;
            int oy = _world.CampCentre.y - 5;

            // Outer wall of a 18x10 compound, in timber.
            Rectangle(ox, oy, 18, 10, WallMaterial.Timber);

            // A cross wall splitting it into two rooms, with a doorway through it.
            for (int y = oy + 1; y < oy + 9; y++)
                Map.Edifices.Set(new Vector2Int(ox + 8, y), EdificeKind.Wall, WallMaterial.Timber);

            Map.Edifices.Set(new Vector2Int(ox + 8, oy + 4), EdificeKind.Door);
            Map.Edifices.Set(new Vector2Int(ox + 4, oy), EdificeKind.Door);

            // A stone pillar: a 2x2 block, so all four corner-filler guards pass at
            // its centre and the patches are visible.
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                    Map.Edifices.Set(new Vector2Int(ox + 12 + dx, oy + 5 + dy), EdificeKind.Wall, WallMaterial.Stone);

            // A short wattle lean-to outside, to show the third material and an
            // open-ended run whose ends must *not* be corner-filled.
            for (int x = ox + 2; x < ox + 7; x++)
                Map.Edifices.Set(new Vector2Int(x, oy - 4), EdificeKind.Wall, WallMaterial.Wattle);

            // Clear the floor inside so pawns have somewhere to stand.
            for (int y = oy + 1; y < oy + 9; y++)
                for (int x = ox + 1; x < ox + 17; x++)
                    if (!Map.Edifices.IsStructure(new Vector2Int(x, y)))
                        Map.Terrain.Set(new Vector2Int(x, y), TerrainKind.Floor);
        }

        private void Rectangle(int x, int y, int width, int height, WallMaterial material)
        {
            for (int i = 0; i < width; i++)
            {
                Map.Edifices.Set(new Vector2Int(x + i, y), EdificeKind.Wall, material);
                Map.Edifices.Set(new Vector2Int(x + i, y + height - 1), EdificeKind.Wall, material);
            }

            for (int i = 0; i < height; i++)
            {
                Map.Edifices.Set(new Vector2Int(x, y + i), EdificeKind.Wall, material);
                Map.Edifices.Set(new Vector2Int(x + width - 1, y + i), EdificeKind.Wall, material);
            }
        }

        /// <summary>
        /// The things the pawns need in order to have anything to do: beds to sleep in,
        /// a larder to eat from, a water butt, and a pond to fall back on when the butt
        /// runs dry. Without these every pawn would want for something it could never
        /// reach, so the job giver would hand out nothing but Wait.
        /// </summary>
        private void PlaceThings()
        {
            int ox = _world.CampCentre.x - 9;
            int oy = _world.CampCentre.y - 5;

            // beds down one side of the first room, one per pawn up to the space
            for (int i = 0; i < Mathf.Min(PawnCount, 6); i++)
                Map.Things.Add(ThingKind.Bed, new Vector2Int(ox + 2 + i, oy + 7));

            Map.Things.Add(ThingKind.FoodStore, new Vector2Int(ox + 3, oy + 2), 24f);
            Map.Things.Add(ThingKind.WaterSource, new Vector2Int(ox + 5, oy + 2), 18f);
            Map.Things.Add(ThingKind.Hearth, new Vector2Int(ox + 4, oy + 4));

            // a pond outside, so thirst has somewhere to go once the butt is empty
            for (int y = oy - 2; y <= oy - 1; y++)
                for (int x = ox + 12; x <= ox + 14; x++)
                    Map.Terrain.Set(new Vector2Int(x, y), TerrainKind.Water);
        }

        private void SpawnPawns()
        {
            int cx = _world.CampCentre.x;
            int cy = _world.CampCentre.y;

            for (int i = 0; i < PawnCount; i++)
            {
                var cell = new Vector2Int(cx - 6 + i, cy - 2);
                var pawn = new Pawn("A" + (i + 1).ToString("00"), Map, cell);

                _pawns.Add(pawn);
                Ticks.Register(pawn);
            }
        }

        private void SetUpCamera()
        {
            Camera camera = Camera.main;

            if (camera == null)
            {
                var go = new GameObject("Main Camera");
                go.tag = "MainCamera";
                camera = go.AddComponent<Camera>();
            }

            camera.orthographic = true;
            // look at the camp rather than the map's middle, which worldgen may well
            // have made a lake; close enough in that a pawn is a readable size, and
            // clamped so the view never hangs off the edge of the world
            const float halfHeight = 14f;
            float halfWidth = halfHeight * camera.aspect;

            camera.orthographicSize = Mathf.Min(halfHeight, Map.Height * 0.5f);
            camera.transform.position = new Vector3(
                Mathf.Clamp(_world.CampCentre.x + 0.5f, halfWidth, Mathf.Max(halfWidth, Map.Width - halfWidth)),
                Mathf.Clamp(_world.CampCentre.y + 0.5f, halfHeight, Mathf.Max(halfHeight, Map.Height - halfHeight)),
                -10f);
            camera.transform.rotation = Quaternion.identity;
            camera.backgroundColor = new Color32(0x3D, 0x4A, 0x2B, 0xFF);
            camera.clearFlags = CameraClearFlags.SolidColor;
        }
    }
}
