using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Agon.Core;

namespace Agon.Tests
{
    /// <summary>
    /// The pathfinder's rules. Every expectation here was first checked against an
    /// independent re-implementation of the same algorithm, so they state the intended
    /// behaviour rather than whatever the code happens to do.
    /// </summary>
    public class PathfindingTests
    {
        /// <summary>
        /// Build a map from rows given top-down, so a fixture reads the way it looks:
        /// '#' wall, '~' water, 'f' forest, '.' open.
        /// </summary>
        private static Map MapFrom(params string[] rowsTopDown)
        {
            int h = rowsTopDown.Length, w = rowsTopDown[0].Length;
            var map = new Map(w, h);
            map.Terrain.Fill(TerrainKind.Soil);

            for (int y = 0; y < h; y++)
            {
                string row = rowsTopDown[h - 1 - y];
                for (int x = 0; x < w; x++)
                {
                    var c = new Vector2Int(x, y);
                    switch (row[x])
                    {
                        case '#': map.Edifices.Set(c, EdificeKind.Wall, WallMaterial.Timber); break;
                        case '~': map.Terrain.Set(c, TerrainKind.Water); break;
                        case 'f': map.Terrain.Set(c, TerrainKind.Forest); break;
                    }
                }
            }

            return map;
        }

        private static List<Vector2Int> Path(Map map, int sx, int sy, int dx, int dy, out bool ok)
        {
            var result = new List<Vector2Int>();
            ok = map.Pathing.FindPath(new Vector2Int(sx, sy), new Vector2Int(dx, dy), result);
            return result;
        }

        [Test]
        public void OpenGround_WalksStraightThere_AndExcludesTheStartCell()
        {
            Map map = MapFrom("........", "........", "........");
            List<Vector2Int> p = Path(map, 0, 1, 7, 1, out bool ok);

            Assert.IsTrue(ok);
            Assert.AreEqual(7, p.Count, "Seven steps to cross seven cells.");
            Assert.AreEqual(new Vector2Int(7, 1), p[p.Count - 1]);
            Assert.AreNotEqual(new Vector2Int(0, 1), p[0], "The pawn already stands on the start cell.");
        }

        [Test]
        public void WallInTheWay_GoesAround()
        {
            Map map = MapFrom(".......",
                              "...#...",
                              "...#...",
                              "...#...",
                              ".......");

            List<Vector2Int> p = Path(map, 0, 2, 6, 2, out bool ok);

            Assert.IsTrue(ok);
            Assert.GreaterOrEqual(p.Count, 6);
            CollectionAssert.DoesNotContain(p, new Vector2Int(3, 2), "The path must not cross the wall.");
            Assert.IsTrue(p.Exists((c) => c.y != 2), "It has to leave the straight line to get round.");
        }

        [Test]
        public void SealedRoom_HasNoPathIn()
        {
            Map map = MapFrom(".......",
                              ".#####.",
                              ".#...#.",
                              ".#####.",
                              ".......");

            Path(map, 0, 0, 3, 2, out bool ok);
            Assert.IsFalse(ok, "Nothing reaches a fully sealed room.");
        }

        [Test]
        public void SingleDoorway_IsUsed_AndSealingItClosesTheRoom()
        {
            Map open = MapFrom(".......",
                               ".#####.",
                               ".#...#.",
                               ".##.##.",
                               ".......");

            List<Vector2Int> p = Path(open, 0, 0, 3, 2, out bool ok);
            Assert.IsTrue(ok, "A one-cell doorway is enough.");
            Assert.IsTrue(p.Contains(new Vector2Int(3, 1)), "The path goes through the doorway cell.");

            Map sealedUp = MapFrom(".......",
                                   ".#####.",
                                   ".#...#.",
                                   ".#####.",
                                   ".......");

            Path(sealedUp, 0, 0, 3, 2, out bool ok2);
            Assert.IsFalse(ok2, "Sealing the doorway shuts the room.");
        }

        [Test]
        public void DiagonalBetweenTwoWallCorners_IsRefused()
        {
            // walls at (0,1) and (1,2); the step (0,2) -> (1,1) would squeeze between them
            Map map = MapFrom(".#.",
                              "#..",
                              "...");

            List<Vector2Int> p = Path(map, 0, 2, 1, 1, out bool ok);

            // either no path at all, or one that goes the long way round rather than
            // through the join — what must never happen is the single diagonal step
            Assert.IsTrue(!ok || p.Count > 1, "A pawn cannot cut the corner between two walls.");
        }

        [Test]
        public void UnwalkableDestination_IsRefused()
        {
            Map wall = MapFrom(".....", "..#..", ".....");
            Path(wall, 0, 0, 2, 1, out bool okWall);
            Assert.IsFalse(okWall, "You cannot path into a wall.");

            Map water = MapFrom(".....", "..~..", ".....");
            Path(water, 0, 0, 2, 1, out bool okWater);
            Assert.IsFalse(okWater, "Nor into open water.");
        }

        [Test]
        public void CostlyTerrain_IsAvoidedWhenThereIsAWayRound()
        {
            Map map = MapFrom(".......",
                              ".fffff.",
                              ".......");

            List<Vector2Int> p = Path(map, 0, 1, 6, 1, out bool ok);
            Assert.IsTrue(ok);

            int forest = 0;
            foreach (Vector2Int c in p)
                if (map.Terrain.At(c) == TerrainKind.Forest) forest++;

            Assert.LessOrEqual(forest, 1, "Forest costs 1.6, so the path should step around it.");
        }

        [Test]
        public void Heuristic_IsOctile_AndNeverOverestimates()
        {
            Assert.AreEqual(0f, PathFinder.Heuristic(Vector2Int.zero, Vector2Int.zero), 1e-5f);
            Assert.AreEqual(1f, PathFinder.Heuristic(Vector2Int.zero, new Vector2Int(1, 0)), 1e-5f);
            Assert.AreEqual(PathFinder.DiagonalCost,
                PathFinder.Heuristic(Vector2Int.zero, new Vector2Int(1, 1)), 1e-5f);

            // three across and one up: one diagonal plus two straight
            Assert.AreEqual(2f + PathFinder.DiagonalCost,
                PathFinder.Heuristic(Vector2Int.zero, new Vector2Int(3, 1)), 1e-5f);

            Map map = MapFrom("..........", "..........", "..........", "..........", "..........");
            for (int i = 0; i < 20; i++)
            {
                var a = new Vector2Int(i % 10, (i * 3) % 5);
                var b = new Vector2Int((i * 7) % 10, (i * 2) % 5);

                var p = new List<Vector2Int>();
                if (!map.Pathing.FindPath(a, b, p))
                    continue;

                float g = 0f;
                Vector2Int at = a;
                foreach (Vector2Int c in p)
                {
                    g += (c.x != at.x && c.y != at.y) ? PathFinder.DiagonalCost : PathFinder.StepCost;
                    at = c;
                }

                Assert.LessOrEqual(PathFinder.Heuristic(a, b), g + 1e-4f,
                    "An inadmissible heuristic would let A* return a path that is not shortest.");
            }
        }

        [Test]
        public void SameCell_IsATrivialSuccess()
        {
            Map map = MapFrom("...", "...", "...");
            List<Vector2Int> p = Path(map, 1, 1, 1, 1, out bool ok);

            Assert.IsTrue(ok);
            Assert.AreEqual(0, p.Count, "Already there: nothing to walk.");
        }

        [Test]
        public void RepeatedSearches_DoNotLeakStateBetweenEachOther()
        {
            Map map = MapFrom(".......",
                              "...#...",
                              "...#...",
                              "...#...",
                              ".......");

            List<Vector2Int> first = Path(map, 0, 2, 6, 2, out bool ok1);
            int firstCount = first.Count;

            for (int i = 0; i < 5; i++)
                Path(map, 0, 0, 6, 4, out bool _);

            List<Vector2Int> again = Path(map, 0, 2, 6, 2, out bool ok2);

            Assert.IsTrue(ok1 && ok2);
            Assert.AreEqual(firstCount, again.Count,
                "The generation stamp must reset each search; a stale closed set would change the answer.");
        }
    }
}
