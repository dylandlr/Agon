using System.Collections.Generic;
using UnityEngine;
using Agon.Core;

namespace Agon.Rendering
{
    /// <summary>
    /// Draws a map's walls. Holds one mesh and one material per wall material, and
    /// rebuilds when the edifice grid changes.
    ///
    /// Rebuilds are coalesced to the end of the frame: a build pass that puts up a
    /// dozen wall segments in one tick would otherwise rebuild the mesh a dozen times.
    /// </summary>
    [AddComponentMenu("Agon/Wall Renderer")]
    public class WallRenderer : MonoBehaviour
    {
        [Tooltip("Shader used for wall meshes. Unlit/Transparent Cutout suits a crisp top-down look.")]
        public Shader WallShader;

        private Map _map;
        private LinkFlags _linkFlags = LinkFlags.WallLike;

        private readonly WallMeshBuilder _builder = new WallMeshBuilder();
        private readonly Dictionary<WallMaterial, Mesh> _meshes = new Dictionary<WallMaterial, Mesh>();
        private readonly Dictionary<WallMaterial, Material> _materials = new Dictionary<WallMaterial, Material>();

        private bool _dirty;

        public void Initialize(Map map, LinkFlags linkFlags = LinkFlags.WallLike)
        {
            if (_map != null)
                _map.Edifices.Changed -= OnEdificeChanged;

            _map = map;
            _linkFlags = linkFlags;
            _map.Edifices.Changed += OnEdificeChanged;

            EnsureMaterials();
            _dirty = true;
        }

        private void OnDestroy()
        {
            if (_map != null)
                _map.Edifices.Changed -= OnEdificeChanged;

            foreach (Mesh mesh in _meshes.Values)
                if (mesh != null) Destroy(mesh);

            foreach (Material material in _materials.Values)
                if (material != null) Destroy(material);
        }

        private void OnEdificeChanged(Vector2Int cell) => _dirty = true;

        private void LateUpdate()
        {
            if (_dirty)
            {
                Rebuild();
                _dirty = false;
            }

            Draw();
        }

        private void EnsureMaterials()
        {
            Shader shader = WallShader != null
                ? WallShader
                : Shader.Find("Unlit/Transparent Cutout");

            if (shader == null)
            {
                Debug.LogError(
                    "WallRenderer: no shader. Assign WallShader, or make sure " +
                    "'Unlit/Transparent Cutout' is included in the build.");
                return;
            }

            for (int m = 0; m < WallMaterialDef.Count; m++)
            {
                var wallMaterial = (WallMaterial)m;

                if (_materials.ContainsKey(wallMaterial))
                    continue;

                Texture2D atlas = WallAtlasGenerator.Generate(wallMaterial);

                var material = new Material(shader)
                {
                    name = "AgonWall_" + wallMaterial,
                    mainTexture = atlas
                };

                if (material.HasProperty("_Cutoff"))
                    material.SetFloat("_Cutoff", 0.5f);

                _materials[wallMaterial] = material;
            }
        }

        private void Rebuild()
        {
            if (_map == null)
                return;

            _builder.Build(_map, _linkFlags, _meshes);
        }

        private void Draw()
        {
            if (_map == null)
                return;

            Matrix4x4 matrix = transform.localToWorldMatrix;

            foreach (KeyValuePair<WallMaterial, Mesh> pair in _meshes)
            {
                Mesh mesh = pair.Value;
                if (mesh == null || mesh.vertexCount == 0)
                    continue;

                if (_materials.TryGetValue(pair.Key, out Material material) && material != null)
                    Graphics.DrawMesh(mesh, matrix, material, gameObject.layer);
            }
        }

        /// <summary>Force a rebuild on the next frame, e.g. after loading a map wholesale.</summary>
        public void SetDirty() => _dirty = true;
    }
}
