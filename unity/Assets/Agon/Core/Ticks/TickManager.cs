using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agon.Core
{
    /// <summary>How often a thing wants ticking — RimWorld's Verse.TickerType.</summary>
    public enum TickerType : byte
    {
        Never = 0,
        Normal = 1,
        Rare = 2,
        Long = 3
    }

    public enum TimeSpeed : byte
    {
        Paused = 0,
        Normal = 1,
        Fast = 2,
        Superfast = 3,
        Ultrafast = 4
    }

    public interface ITickable
    {
        TickerType TickerType { get; }
        void Tick();
    }

    /// <summary>
    /// Drives simulation time. Modelled on RimWorld's Verse.TickManager: a tick
    /// counter advanced by a speed multiplier, with three ticker lists so that most
    /// things are visited rarely.
    ///
    /// Rare ticks fire every 250 ticks and long ticks every 2000, as in RimWorld;
    /// the lists are spread across those windows by index so the work per tick stays
    /// level instead of spiking on one tick in 250.
    /// </summary>
    public class TickManager
    {
        public const int TicksPerRare = 250;
        public const int TicksPerLong = 2000;

        /// <summary>Ticks in one in-game day. Agon's HTML build uses 50; kept identical here.</summary>
        public const int TicksPerDay = 50;

        /// <summary>Ticks in one in-game year (four seasons of 500).</summary>
        public const int TicksPerYear = 2000;

        private readonly List<ITickable> _normal = new List<ITickable>();
        private readonly List<ITickable> _rare = new List<ITickable>();
        private readonly List<ITickable> _long = new List<ITickable>();

        private float _accumulator;

        public int TicksGame { get; private set; }

        public TimeSpeed CurTimeSpeed { get; set; } = TimeSpeed.Normal;

        /// <summary>Raised once per tick, after every registered tickable has ticked.</summary>
        public event Action<int> Ticked;

        public float TickRateMultiplier
        {
            get
            {
                switch (CurTimeSpeed)
                {
                    case TimeSpeed.Paused: return 0f;
                    case TimeSpeed.Normal: return 1f;
                    case TimeSpeed.Fast: return 3f;
                    case TimeSpeed.Superfast: return 6f;
                    case TimeSpeed.Ultrafast: return 15f;
                    default: return 1f;
                }
            }
        }

        /// <summary>Seconds of real time one tick takes at the current speed; 0 when paused.</summary>
        public float SecondsPerTick
        {
            get
            {
                float rate = TickRateMultiplier;
                return rate == 0f ? 0f : 1f / (60f * rate);
            }
        }

        public void Register(ITickable t)
        {
            switch (t.TickerType)
            {
                case TickerType.Normal: _normal.Add(t); break;
                case TickerType.Rare: _rare.Add(t); break;
                case TickerType.Long: _long.Add(t); break;
            }
        }

        public void Deregister(ITickable t)
        {
            _normal.Remove(t);
            _rare.Remove(t);
            _long.Remove(t);
        }

        /// <summary>
        /// Advance by elapsed real time, running as many ticks as the current speed
        /// calls for. <paramref name="maxTicksPerFrame"/> caps the catch-up after a
        /// stall so a long frame cannot spiral.
        /// </summary>
        public void Update(float deltaTime, int maxTicksPerFrame = 20)
        {
            float secondsPerTick = SecondsPerTick;
            if (secondsPerTick <= 0f)
                return;

            _accumulator += deltaTime;

            int ran = 0;
            while (_accumulator >= secondsPerTick && ran < maxTicksPerFrame)
            {
                _accumulator -= secondsPerTick;
                DoSingleTick();
                ran++;
            }

            if (ran >= maxTicksPerFrame)
                _accumulator = 0f;
        }

        public void DoSingleTick()
        {
            for (int i = 0; i < _normal.Count; i++)
                _normal[i].Tick();

            // Spread the rare and long lists across their windows: a thing ticks on the
            // tick whose remainder matches its own index.
            TickSpread(_rare, TicksPerRare);
            TickSpread(_long, TicksPerLong);

            TicksGame++;
            Ticked?.Invoke(TicksGame);
        }

        private void TickSpread(List<ITickable> list, int interval)
        {
            if (list.Count == 0)
                return;

            int phase = TicksGame % interval;
            for (int i = phase; i < list.Count; i += interval)
                list[i].Tick();
        }

        public int DayOfYear => (TicksGame / TicksPerDay) % (TicksPerYear / TicksPerDay);

        public int Year => TicksGame / TicksPerYear;

        /// <summary>Season index, 0..3, each 500 ticks long.</summary>
        public int Season => (TicksGame % TicksPerYear) / (TicksPerYear / 4);

        /// <summary>Position within the day, 0..1, where 0 is midnight.</summary>
        public float DayPercent => (TicksGame % TicksPerDay) / (float)TicksPerDay;
    }
}
