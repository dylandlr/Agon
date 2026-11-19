using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace Agon.Eos
{
    [Serializable] public class EosAgent
    {
        public int slot, row, column, age_ticks;
        public bool alive, existed;
        public float satiety, hydration, energy, health, body_temp, stress, injury;
        public float[] inventory;
    }
    [Serializable] public class EosFields
    {
        public int[] terrain;
        public float[] height, plants, meat, cooked, wood, stones, fire, snow;
    }
    [Serializable] public class TerrainIds { public int grass, forest, rock, water; }
    [Serializable] public class EosSnapshot
    {
        public int tick, grid_size, weather;
        public float tick_minutes;
        public string policy, config_id, engine_version, learning;
        public EosFields fields;
        public EosAgent[] agents;
        public TerrainIds terrain_ids;
    }
    [Serializable] class BridgeError { public string type, message; }
    [Serializable] class BridgeResponse
    {
        public int version, id;
        public bool ok;
        public EosSnapshot result;
        public BridgeError error;
    }

    /// <summary>Presentation only: every simulation tick runs in the Eos child process.</summary>
    public sealed class EosClient : MonoBehaviour
    {
        public string PythonExecutable;
        public string ServerScript;
        public string EngineRoot;
        public string TrainedRun;
        public string SaveDirectory;
        public bool DiagnosticRandom = true;
        public bool Paused = true;
        [Range(1, 16)] public int StepsPerRequest = 1;
        public EosSnapshot Snapshot { get; private set; }
        public string Status { get; private set; } = "Not started";
        public string Failure { get; private set; }
        public bool Busy => _pending != null;
        Process _process;
        Task<string> _pending;
        int _request;
        float _sentAt;
        Texture2D _terrain, _white;
        Sprite _mapSprite, _pawnSprite;
        SpriteRenderer _map;
        readonly Dictionary<int, SpriteRenderer> _pawns = new Dictionary<int, SpriteRenderer>();

        void Start()
        {
            try
            {
                var root = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
                if (string.IsNullOrEmpty(PythonExecutable))
                    PythonExecutable = Path.Combine(root, "node_modules/.cache/eos-venv/Scripts/python.exe");
                if (string.IsNullOrEmpty(ServerScript)) ServerScript = Path.Combine(root, "bridge/eos_server.py");
                if (!DiagnosticRandom && string.IsNullOrWhiteSpace(TrainedRun))
                    throw new InvalidOperationException("Select a trained run; no fallback policy is permitted.");
                var arguments = Quote(ServerScript);
                if (!string.IsNullOrEmpty(EngineRoot)) arguments += " --engine-root " + Quote(EngineRoot);
                if (!string.IsNullOrEmpty(SaveDirectory)) arguments += " --save-dir " + Quote(SaveDirectory);
                if (!DiagnosticRandom) arguments += " --run " + Quote(TrainedRun);
                _process = new Process { StartInfo = new ProcessStartInfo(PythonExecutable, arguments)
                {
                    UseShellExecute = false, CreateNoWindow = true,
                    RedirectStandardInput = true, RedirectStandardOutput = true,
                    RedirectStandardError = true, WorkingDirectory = root
                }};
                // Drain diagnostics so a full pipe can never stall simulation.
                _process.ErrorDataReceived += (_, e) => { };
                _process.Start();
                _process.BeginErrorReadLine();
                Send("\"command\":\"reset\",\"seed\":7,\"mode\":\"" +
                     (DiagnosticRandom ? "random\",\"config\":{\"grid_size\":24,\"n_agents\":4}" : "trained\"") );
                Status = "Starting Eos (first JAX compilation may take a minute)";
            }
            catch (Exception e) { Fail(e.Message); }
        }

        static string Quote(string value)
        {
            if (value.Contains("\"") || value.Contains("\n")) throw new ArgumentException("Invalid path");
            return "\"" + value + "\"";
        }

        void Send(string body)
        {
            if (Busy) throw new InvalidOperationException("A simulation request is already pending");
            _process.StandardInput.WriteLine("{\"version\":1,\"id\":" + (++_request) + "," + body + "}");
            _process.StandardInput.Flush();
            _pending = _process.StandardOutput.ReadLineAsync();
            _sentAt = Time.realtimeSinceStartup;
        }

        public void StepOnce()
        {
            if (Busy || Snapshot == null || Failure != null) return;
            try { Send("\"command\":\"step\",\"count\":" + Mathf.Clamp(StepsPerRequest, 1, 16)); }
            catch (Exception e) { Fail(e.Message); }
        }

        public void SaveGame() => SaveCommand("save");
        public void LoadGame() => SaveCommand("load");
        void SaveCommand(string command)
        {
            if (Busy || Snapshot == null) return;
            Paused = true;
            Failure = null;
            try { Send("\"command\":\"" + command + "\",\"slot\":\"quicksave\""); }
            catch (Exception e) { Fail(e.Message); }
        }

        void Update()
        {
            if (Failure != null) return;
            if (_pending != null)
            {
                if (!_pending.IsCompleted)
                {
                    if (Time.realtimeSinceStartup - _sentAt > 180) Fail("Eos request timed out; simulation paused.");
                    return;
                }
                var pending = _pending;
                _pending = null;
                try
                {
                    var line = pending.GetAwaiter().GetResult();
                    if (line == null) throw new IOException("Eos exited before responding.");
                    var response = JsonUtility.FromJson<BridgeResponse>(line);
                    if (response.version != 1 || response.id != _request) throw new IOException("Protocol response mismatch");
                    if (!response.ok) throw new IOException(response.error?.message ?? "Eos request failed");
                    Snapshot = response.result;
                    RenderSnapshot();
                    Status = Snapshot.policy + " | tick " + Snapshot.tick;
                }
                catch (Exception e) { Fail(e.Message); }
            }
            if (!Paused) StepOnce();
        }

        void RenderSnapshot()
        {
            int size = Snapshot.grid_size;
            if (_terrain == null)
            {
                _terrain = new Texture2D(size, size, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
                _mapSprite = Sprite.Create(_terrain, new Rect(0, 0, size, size), Vector2.zero, 1);
                _map = new GameObject("Eos terrain").AddComponent<SpriteRenderer>();
                _map.transform.SetParent(transform, false);
                _map.sprite = _mapSprite;
                _map.sortingOrder = -10;
                _white = new Texture2D(1, 1);
                _white.SetPixel(0, 0, Color.white); _white.Apply();
                _pawnSprite = Sprite.Create(_white, new Rect(0, 0, 1, 1), new Vector2(.5f, .5f), 1);
                Camera camera = Camera.main;
                if (camera == null)
                {
                    camera = new GameObject("Eos Camera").AddComponent<Camera>();
                    camera.tag = "MainCamera";
                    camera.transform.SetParent(transform, false);
                }
                camera.orthographic = true;
                camera.orthographicSize = size * .55f;
                camera.transform.position = new Vector3(size / 2f, size / 2f, -10);
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(.06f, .08f, .09f);
            }
            var pixels = new Color[size * size];
            for (int row = 0; row < size; row++) for (int col = 0; col < size; col++)
            {
                int index = row * size + col, terrain = Snapshot.fields.terrain[index];
                Color color = terrain == Snapshot.terrain_ids.water ? new Color(.16f,.36f,.48f) :
                    terrain == Snapshot.terrain_ids.rock ? new Color(.43f,.42f,.39f) :
                    terrain == Snapshot.terrain_ids.forest ? new Color(.18f,.32f,.2f) : new Color(.39f,.47f,.27f);
                if (Snapshot.fields.fire[index] > 0) color = new Color(1f,.36f,.08f);
                pixels[(size-1-row)*size+col] = color;
            }
            _terrain.SetPixels(pixels); _terrain.Apply();
            foreach (var agent in Snapshot.agents)
            {
                if (!_pawns.TryGetValue(agent.slot, out var pawn))
                {
                    pawn = new GameObject("Eos agent " + agent.slot).AddComponent<SpriteRenderer>();
                    pawn.transform.SetParent(transform, false); pawn.sprite = _pawnSprite;
                    pawn.transform.localScale = new Vector3(.55f, .7f, 1);
                    _pawns.Add(agent.slot, pawn);
                }
                pawn.transform.position = new Vector3(agent.column+.5f, size-1-agent.row+.5f, -1);
                pawn.enabled = agent.existed;
                pawn.color = agent.alive ? new Color(.95f,.79f,.5f) : Color.gray;
            }
        }

        void Fail(string message) { Failure = message; Status = message; Paused = true; }
        void OnDestroy()
        {
            if (_process != null)
            {
                try { if (!_process.HasExited) _process.Kill(); } catch (InvalidOperationException) { }
                _process.Dispose();
            }
            if (_mapSprite != null) Destroy(_mapSprite);
            if (_pawnSprite != null) Destroy(_pawnSprite);
            if (_terrain != null) Destroy(_terrain);
            if (_white != null) Destroy(_white);
        }
    }
}
