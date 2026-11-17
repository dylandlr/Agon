using NUnit.Framework;
using UnityEngine;
using Agon.Core;

namespace Agon.Tests
{
    public class TickManagerTests
    {
        private class Counter : ITickable
        {
            public int Ticks;
            public TickerType TickerType { get; }

            public Counter(TickerType type) { TickerType = type; }

            public void Tick() => Ticks++;
        }

        [Test]
        public void NormalTickers_TickEveryTick()
        {
            var ticks = new TickManager();
            var counter = new Counter(TickerType.Normal);
            ticks.Register(counter);

            for (int i = 0; i < 10; i++)
                ticks.DoSingleTick();

            Assert.AreEqual(10, counter.Ticks);
            Assert.AreEqual(10, ticks.TicksGame);
        }

        [Test]
        public void RareTickers_TickOncePerWindow_AndAreSpreadAcrossIt()
        {
            var ticks = new TickManager();
            var counters = new Counter[3];

            for (int i = 0; i < counters.Length; i++)
            {
                counters[i] = new Counter(TickerType.Rare);
                ticks.Register(counters[i]);
            }

            for (int i = 0; i < TickManager.TicksPerRare; i++)
                ticks.DoSingleTick();

            foreach (Counter c in counters)
                Assert.AreEqual(1, c.Ticks, "Each rare ticker fires exactly once per window.");
        }

        [Test]
        public void RareTickers_DoNotAllFireOnTheSameTick()
        {
            var ticks = new TickManager();
            var a = new Counter(TickerType.Rare);
            var b = new Counter(TickerType.Rare);
            ticks.Register(a);
            ticks.Register(b);

            ticks.DoSingleTick();

            Assert.AreEqual(1, a.Ticks);
            Assert.AreEqual(0, b.Ticks, "The second rare ticker waits for its own phase.");
        }

        [Test]
        public void Paused_RunsNoTicks()
        {
            var ticks = new TickManager { CurTimeSpeed = TimeSpeed.Paused };
            var counter = new Counter(TickerType.Normal);
            ticks.Register(counter);

            ticks.Update(1f);

            Assert.AreEqual(0, counter.Ticks);
            Assert.AreEqual(0f, ticks.SecondsPerTick);
        }

        [Test]
        public void FasterSpeeds_RunMoreTicksForTheSameElapsedTime()
        {
            int TicksIn(TimeSpeed speed, float seconds)
            {
                var ticks = new TickManager { CurTimeSpeed = speed };
                var counter = new Counter(TickerType.Normal);
                ticks.Register(counter);
                ticks.Update(seconds, maxTicksPerFrame: 10000);
                return counter.Ticks;
            }

            int normal = TicksIn(TimeSpeed.Normal, 1f);
            int fast = TicksIn(TimeSpeed.Fast, 1f);

            // Tolerances are here because the accumulator is a float: subtracting
            // 1/60 six hundred times drifts by less than one tick, but not by zero.
            Assert.That(normal, Is.EqualTo(60).Within(1), "Normal speed is 60 ticks a second.");
            Assert.That(fast, Is.EqualTo(180).Within(2), "Fast is three times normal.");
            Assert.Greater(fast, normal * 2, "Fast must be clearly faster than normal.");
        }

        [Test]
        public void CatchUp_IsCapped_SoAStallCannotSpiral()
        {
            var ticks = new TickManager();
            var counter = new Counter(TickerType.Normal);
            ticks.Register(counter);

            ticks.Update(10f, maxTicksPerFrame: 20);

            Assert.AreEqual(20, counter.Ticks, "A long frame runs at most the cap.");

            ticks.Update(0f);
            Assert.AreEqual(20, counter.Ticks, "The backlog is dropped, not carried.");
        }

        [Test]
        public void Calendar_DerivesFromTheTickCount()
        {
            var ticks = new TickManager();

            for (int i = 0; i < TickManager.TicksPerYear; i++)
                ticks.DoSingleTick();

            Assert.AreEqual(1, ticks.Year);
            Assert.AreEqual(0, ticks.Season, "A full year wraps back to the first season.");
        }
    }

    public class NeedsTests
    {
        private static Pawn MakePawn()
        {
            var map = new Map(8, 8);
            map.Terrain.Fill(TerrainKind.Soil);
            return new Pawn("A01", map, new Vector2Int(4, 4));
        }

        [Test]
        public void Needs_FallOverTime_AndClampAtZero()
        {
            Pawn pawn = MakePawn();
            float start = pawn.Needs.Food.CurLevel;

            pawn.Needs.NeedsTrackerTick();
            Assert.Less(pawn.Needs.Food.CurLevel, start);

            for (int i = 0; i < 10000; i++)
                pawn.Needs.NeedsTrackerTick();

            Assert.AreEqual(0f, pawn.Needs.Food.CurLevel, "A need never goes below zero.");
        }

        [Test]
        public void Rest_RecoversWhileAsleep()
        {
            Pawn pawn = MakePawn();
            pawn.Needs.Rest.CurLevel = 0.2f;
            pawn.Asleep = true;

            pawn.Needs.NeedsTrackerTick();

            Assert.Greater(pawn.Needs.Rest.CurLevel, 0.2f);
        }

        [Test]
        public void Cold_MakesHungerFallFaster()
        {
            Pawn warm = MakePawn();
            Pawn cold = MakePawn();
            cold.Comfort = -20f;

            warm.Needs.NeedsTrackerTick();
            cold.Needs.NeedsTrackerTick();

            Assert.Less(cold.Needs.Food.CurLevel, warm.Needs.Food.CurLevel,
                "A cold pawn burns food to hold its temperature.");
        }

        [Test]
        public void MostUrgent_PicksTheNeedFurthestPastItsOwnThreshold()
        {
            Pawn pawn = MakePawn();

            Assert.IsNull(pawn.Needs.MostUrgent(), "Nothing is urgent when every need is high.");

            pawn.Needs.Food.CurLevel = 0.39f;   // just under its 0.40 threshold
            pawn.Needs.Water.CurLevel = 0.10f;  // far under its 0.45 threshold

            Assert.AreSame(pawn.Needs.Water, pawn.Needs.MostUrgent());
        }

        [Test]
        public void StarvingPawn_LosesHealth_AndEventuallyDies()
        {
            Pawn pawn = MakePawn();
            pawn.Needs.Food.CurLevel = 0f;
            pawn.Needs.Water.CurLevel = 0f;

            for (int i = 0; i < 1000 && pawn.Alive; i++)
                pawn.Tick();

            Assert.IsFalse(pawn.Alive, "A pawn with no food or water dies.");
            Assert.AreEqual(0f, pawn.Health);
        }
    }

    public class MapTests
    {
        [Test]
        public void CellAndIndex_RoundTrip()
        {
            var map = new Map(13, 7);

            for (int i = 0; i < map.CellCount; i++)
                Assert.AreEqual(i, map.CellToIndex(map.IndexToCell(i)));
        }

        [Test]
        public void Walkable_IsFalseThroughWalls_AndTrueThroughDoors()
        {
            var map = new Map(8, 8);
            map.Terrain.Fill(TerrainKind.Soil);

            map.Edifices.Set(new Vector2Int(3, 3), EdificeKind.Wall, WallMaterial.Timber);
            map.Edifices.Set(new Vector2Int(4, 3), EdificeKind.Door);

            Assert.IsFalse(map.Terrain.Walkable(new Vector2Int(3, 3)));
            Assert.IsTrue(map.Terrain.Walkable(new Vector2Int(4, 3)));
            Assert.IsFalse(map.Terrain.Walkable(new Vector2Int(-1, 3)), "Off-map is not walkable.");
        }

        [Test]
        public void EdificeChange_RaisesChanged_SoRenderersCanRebuild()
        {
            var map = new Map(8, 8);
            int changes = 0;
            map.Edifices.Changed += _ => changes++;

            map.Edifices.Set(new Vector2Int(2, 2), EdificeKind.Wall, WallMaterial.Stone);
            Assert.AreEqual(1, changes);

            map.Edifices.Set(new Vector2Int(2, 2), EdificeKind.Wall, WallMaterial.Stone);
            Assert.AreEqual(1, changes, "Setting the same thing again changes nothing.");
        }
    }
}
