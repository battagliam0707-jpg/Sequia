import os
import zipfile
import json
import base64
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import zarr
import icechunk
import geopandas as gpd
from shapely.ops import unary_union
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. DATOS DE CENTRALES HIDROELÉCTRICAS
# ==========================================
centrales_data = [
    {"id": 1,  "nombre": "Itaipú",            "pais": "Brasil / Paraguay",    "cuenca": "Río Paraná",        "capacidad_mw": 14000, "nivel_embalse": 92, "lon": -54.5889, "lat": -25.4081},
    {"id": 2,  "nombre": "Belo Monte",        "pais": "Brasil",               "cuenca": "Río Xingu",         "capacidad_mw": 11233, "nivel_embalse": 74, "lon": -51.7750, "lat": -3.1264},
    {"id": 3,  "nombre": "Guri",              "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 10235, "nivel_embalse": 84, "lon": -62.9981, "lat": 7.7650},
    {"id": 4,  "nombre": "Tucuruí",           "pais": "Brasil",               "cuenca": "Río Tocantins",     "capacidad_mw": 8370,  "nivel_embalse": 79, "lon": -49.6522, "lat": -3.8319},
    {"id": 5,  "nombre": "Paulo Afonso",      "pais": "Brasil",               "cuenca": "Río São Francisco", "capacidad_mw": 4279,  "nivel_embalse": 58, "lon": -38.2019, "lat": -9.3986},
    {"id": 6,  "nombre": "Jirau",             "pais": "Brasil",               "cuenca": "Río Madeira",       "capacidad_mw": 3750,  "nivel_embalse": 71, "lon": -64.6458, "lat": -9.2669},
    {"id": 7,  "nombre": "Santo Antônio",     "pais": "Brasil",               "cuenca": "Río Madeira",       "capacidad_mw": 3568,  "nivel_embalse": 72, "lon": -63.9506, "lat": -8.8017},
    {"id": 8,  "nombre": "Ilha Solteira",     "pais": "Brasil",               "cuenca": "Río Paraná",        "capacidad_mw": 3444,  "nivel_embalse": 77, "lon": -51.3636, "lat": -20.3822},
    {"id": 9,  "nombre": "Yacyretá",          "pais": "Argentina / Paraguay", "cuenca": "Río Paraná",        "capacidad_mw": 3200,  "nivel_embalse": 91, "lon": -56.7331, "lat": -27.4831},
    {"id": 10, "nombre": "Macagua",           "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 3167,  "nivel_embalse": 85, "lon": -62.6683, "lat": 8.3042},
    {"id": 11, "nombre": "Xingó",             "pais": "Brasil",               "cuenca": "Río São Francisco", "capacidad_mw": 3162,  "nivel_embalse": 61, "lon": -37.7936, "lat": -9.6206},
    {"id": 12, "nombre": "Hidroituango",      "pais": "Colombia",             "cuenca": "Río Cauca",         "capacidad_mw": 2400,  "nivel_embalse": 82, "lon": -75.6672, "lat": 7.1333},
    {"id": 13, "nombre": "Caruachi",          "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 2160,  "nivel_embalse": 83, "lon": -62.7983, "lat": 8.1597},
    {"id": 14, "nombre": "Salto Grande",      "pais": "Argentina / Uruguay",  "cuenca": "Río Uruguay",       "capacidad_mw": 1890,  "nivel_embalse": 76, "lon": -57.9394, "lat": -31.2744},
    {"id": 15, "nombre": "Piedra del Águila", "pais": "Argentina",            "cuenca": "Río Limay",         "capacidad_mw": 1400,  "nivel_embalse": 64, "lon": -69.9922, "lat": -40.1889},
    {"id": 16, "nombre": "Del Guavio",        "pais": "Colombia",             "cuenca": "Río Guavio",        "capacidad_mw": 1213,  "nivel_embalse": 69, "lon": -73.4833, "lat": 4.7000},
    {"id": 17, "nombre": "El Chocón",         "pais": "Argentina",            "cuenca": "Río Limay",         "capacidad_mw": 1200,  "nivel_embalse": 23, "lon": -68.7611, "lat": -39.2653},
    {"id": 18, "nombre": "Mantaro",           "pais": "Perú",                 "cuenca": "Río Mantaro",       "capacidad_mw": 1008,  "nivel_embalse": 53, "lon": -74.6819, "lat": -12.3592},
    {"id": 19, "nombre": "Ralco",             "pais": "Chile",                "cuenca": "Río Biobío",        "capacidad_mw": 690,   "nivel_embalse": 21, "lon": -71.4822, "lat": -38.0439},
    {"id": 20, "nombre": "Guatapé",           "pais": "Colombia",             "cuenca": "Río Nare",          "capacidad_mw": 560,   "nivel_embalse": 67, "lon": -75.1583, "lat": 6.2333}
]

represas = gpd.GeoDataFrame(
    centrales_data,
    geometry=gpd.points_from_xy([c["lon"] for c in centrales_data], [c["lat"] for c in centrales_data]),
    crs="EPSG:4326"
)

# ==========================================
# 2. CONEXIÓN A ECMWF SEAS5 (FORECAST + HINDCAST)
# ==========================================
print("Conectando a AWS S3 (ECMWF SEAS5)...")
bucket_name = "planette-c3s-seasonal-forecasts"
resp_head = requests.head(f"https://{bucket_name}.s3.amazonaws.com/", timeout=30)
region_bucket = resp_head.headers.get("x-amz-bucket-region", "us-east-2")

def abrir_dataset_icechunk(prefix):
    storage = icechunk.s3_storage(bucket=bucket_name, prefix=prefix, region=region_bucket, anonymous=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    root = zarr.open_group(session.store, mode="r")
    subgrupos = sorted(list(root.group_keys()))
    if subgrupos:
        cand = [g for g in subgrupos if any(k in g.lower() for k in ["tprate", "tp", "precip", "ro"])]
        g_obj = cand[0] if cand else subgrupos[-1]
        ds_out = xr.open_zarr(session.store, group=g_obj, consolidated=False)
        if not list(ds_out.data_vars):
            sub_anid = sorted(list(root[g_obj].group_keys()))
            cand_a = [g for g in sub_anid if any(k in g.lower() for k in ["tprate", "tp", "precip", "ro"])]
            ds_out = xr.open_zarr(session.store, group=f"{g_obj}/{cand_a[0] if cand_a else sub_anid[-1]}", consolidated=False)
        return ds_out
    return xr.open_zarr(session.store, consolidated=False)

ds_fc = abrir_dataset_icechunk("seas5_ic/forecast")
ds_hc = abrir_dataset_icechunk("seas5_ic/hindcast")

var_fc = next((v for v in ds_fc.data_vars if any(k in v.lower() for k in ["tprate", "tp", "precip", "ro"])), list(ds_fc.data_vars)[0])
var_hc = next((v for v in ds_hc.data_vars if any(k in v.lower() for k in ["tprate", "tp", "precip", "ro"])), list(ds_hc.data_vars)[0])

def preparar_sudamerica(da):
    renombres = {}
    for c in da.coords:
        cl = c.lower()
        if cl in ["latitude", "lat"]: renombres[c] = "y"
        elif cl in ["longitude", "lon"]: renombres[c] = "x"
    da = da.rename(renombres)
    if float(da.x.max()) > 180:
        da = da.assign_coords(x=(((da.x + 180) % 360) - 180)).sortby("x")
    if float(da.y[0]) < float(da.y[-1]):
        da = da.sortby("y", ascending=False)
    return da.sel(x=slice(-85, -30), y=slice(15, -60))

da_fc = preparar_sudamerica(ds_fc[var_fc])
da_hc = preparar_sudamerica(ds_hc[var_hc])

dim_init_fc = next((d for d in da_fc.dims if d not in ["x", "y"] and any(k in d.lower() for k in ["init", "time", "start", "ref"])), None)
fecha_emision_str = datetime.now(timezone.utc).strftime("%Y-%m")

if dim_init_fc and dim_init_fc in da_fc.dims and da_fc.sizes[dim_init_fc] > 1:
    dim_step_fc = next((d for d in da_fc.dims if d not in ["x", "y", dim_init_fc] and any(k in d.lower() for k in ["step", "lead", "valid", "forecast"])), None)
    if dim_step_fc:
        ultima_fecha = pd.Timestamp(da_fc[dim_init_fc].values[-1])
        mes_emision = ultima_fecha.month
        fecha_emision_str = ultima_fecha.strftime("%Y-%m-%d")
        da_fc_sel = da_fc.isel({dim_init_fc: -1, dim_step_fc: slice(0, 6)})
    else:
        ultima_fecha = pd.Timestamp(da_fc[dim_init_fc].values[0])
        mes_emision = ultima_fecha.month
        fecha_emision_str = ultima_fecha.strftime("%Y-%m-%d")
        da_fc_sel = da_fc.isel({dim_init_fc: slice(0, 6)})
else:
    dim_seq = next((d for d in da_fc.dims if d not in ["x", "y", "member", "number", "realization"]), None)
    if dim_seq and np.issubdtype(da_fc[dim_seq].dtype, np.datetime64):
        ultima_fecha = pd.Timestamp(da_fc[dim_seq].values[0])
        mes_emision = ultima_fecha.month
        fecha_emision_str = ultima_fecha.strftime("%Y-%m-%d")
    else:
        mes_emision = datetime.now(timezone.utc).month
    da_fc_sel = da_fc.isel({dim_seq: slice(0, 6)}) if dim_seq else da_fc

fc_6m = da_fc_sel.mean(dim=[d for d in da_fc_sel.dims if d not in ["x", "y"]], skipna=True).compute()

dim_init_hc = next((d for d in da_hc.dims if d not in ["x", "y"] and any(k in d.lower() for k in ["init", "start", "ref", "time"])), None)
dim_step_hc = next((d for d in da_hc.dims if d not in ["x", "y", dim_init_hc] and any(k in d.lower() for k in ["step", "lead", "valid", "forecast"])), None)

da_hc_sel = da_hc
if dim_init_hc and np.issubdtype(da_hc[dim_init_hc].dtype, np.datetime64):
    meses_hc = pd.DatetimeIndex(da_hc[dim_init_hc].values).month
    if (meses_hc == mes_emision).any():
        da_hc_sel = da_hc_sel.isel({dim_init_hc: (meses_hc == mes_emision)})
if dim_step_hc:
    da_hc_sel = da_hc_sel.isel({dim_step_hc: slice(0, 6)})

hc_6m = da_hc_sel.mean(dim=[d for d in da_hc_sel.dims if d not in ["x", "y"]], skipna=True).compute()
hc_6m = hc_6m.interp_like(fc_6m, method="nearest")

anomalia_pct = ((fc_6m - hc_6m) / (xr.where(hc_6m > 1e-9, hc_6m, np.nan))) * 100.0
anomalia_pct = anomalia_pct.clip(-80.0, 80.0).rio.write_crs("EPSG:4326")
anomalia_pct.rio.to_raster("anomalia_seas5_6m_sa.tif")

# ==========================================
# 3. CUENCAS AGUAS ARRIBA (HYDROBASINS)
# ==========================================
print("Procesando cuencas HydroBASINS...")
shp_cuencas = "hybas_sa_lev06_v1c.shp"
if not os.path.exists(shp_cuencas):
    resp = requests.get("https://data.hydrosheds.org/file/hydrobasins/standard/hybas_sa_lev06_v1c.zip", headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
    resp.raise_for_status()
    with open("hybas_sa_lev06_v1c.zip", "wb") as f: f.write(resp.content)
    with zipfile.ZipFile("hybas_sa_lev06_v1c.zip", "r") as z: z.extractall(".")

cuencas = gpd.read_file(shp_cuencas).to_crs("EPSG:4326")
aguas_arriba_map = {}
for _, row in cuencas[["HYBAS_ID", "NEXT_DOWN"]].iterrows():
    aguas_arriba_map.setdefault(row["NEXT_DOWN"], []).append(row["HYBAS_ID"])

def obtener_subcuencas(hybas_id):
    visitadas, pila = set(), [hybas_id]
    while pila:
        act = pila.pop()
        if act not in visitadas:
            visitadas.add(act)
            pila.extend(aguas_arriba_map.get(act, []))
    return list(visitadas)

represas_cuenca = gpd.sjoin_nearest(represas.to_crs("EPSG:3857"), cuencas[["HYBAS_ID", "geometry"]].to_crs("EPSG:3857"), how="left").to_crs("EPSG:4326")

poligonos, num_sub = [], []
for _, row in represas_cuenca.iterrows():
    ids = obtener_subcuencas(row["HYBAS_ID"])
    num_sub.append(len(ids))
    poligonos.append(unary_union(cuencas[cuencas["HYBAS_ID"].isin(ids)].geometry))

gdf_captacion = gpd.GeoDataFrame(geometry=poligonos, crs="EPSG:4326")
stats = zonal_stats(gdf_captacion, "anomalia_seas5_6m_sa.tif", stats=["mean"], all_touched=True)

LIMITE = 35.0
anomalias_cuenca, riesgo_climatico, riesgo_operativo = [], [], []
for idx, s in enumerate(stats):
    anom = round(float(s["mean"]), 1) if s["mean"] is not None else 0.0
    anomalias_cuenca.append(anom)
    r_clima = round(float(np.clip(((LIMITE - anom) / (2.0 * LIMITE)) * 100.0, 0, 100)), 1)
    riesgo_climatico.append(r_clima)
    vuln_embalse = 100.0 - float(represas.loc[idx, "nivel_embalse"])
    riesgo_operativo.append(round((0.65 * r_clima) + (0.35 * vuln_embalse), 1))

represas["subcuencas_aporte"] = num_sub
represas["anomalia_pct"] = anomalias_cuenca
represas["riesgo_climatico"] = riesgo_climatico
represas["riesgo_operativo"] = riesgo_operativo

# ==========================================
# 4. RENDERIZAR PNG Y COMPILAR index.html
# ==========================================
da_r = rioxarray.open_rasterio("anomalia_seas5_6m_sa.tif").squeeze()
min_lon, min_lat, max_lon, max_lat = da_r.rio.bounds()
bounds_leaflet = [[float(min_lat), float(min_lon)], [float(max_lat), float(max_lon)]]

vals = da_r.values.astype(float)
norm = np.clip((LIMITE - vals) / (2.0 * LIMITE), 0.0, 1.0)
cmap = LinearSegmentedColormap.from_list("anom_sa", ["#1d4ed8", "#10b981", "#fef08a", "#f97316", "#b91c1c"], N=256)
rgba = cmap(norm)
rgba[..., 3] = np.where(np.isnan(vals), 0.0, 0.76)

plt.imsave("mapa_anomalia_sa.png", rgba)
with open("mapa_anomalia_sa.png", "rb") as f:
    png_b64 = base64.b64encode(f.read()).decode("utf-8")

cuencas_export = gdf_captacion.copy()
cuencas_export["nombre"] = represas["nombre"].values
cuencas_export["geometry"] = cuencas_export.geometry.simplify(0.02, preserve_topology=True)

actualizado_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

html_template = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Visor Anomalía de Sequía y Riesgo Hidroeléctrico (ECMWF SEAS5)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
    body { display: flex; height: 100vh; background: #0f172a; color: #f8fafc; overflow: hidden; }
    #sidebar { width: 400px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; z-index: 1000; }
    .header { padding: 14px 18px; background: #0f172a; border-bottom: 1px solid #334155; }
    .header h1 { font-size: 16px; color: #38bdf8; }
    .header p { font-size: 11.5px; color: #94a3b8; margin-top: 3px; }
    .meta-tag { display: inline-block; margin-top: 6px; padding: 2px 8px; background: #1e293b; border: 1px solid #334155; border-radius: 4px; font-size: 10.5px; color: #38bdf8; }
    .controls { padding: 12px 18px; border-bottom: 1px solid #334155; font-size: 13px; }
    .mode-box { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; background: #0f172a; padding: 4px; border-radius: 8px; border: 1px solid #334155; }
    .mode-btn { padding: 7px 6px; border: none; border-radius: 6px; font-size: 11.5px; font-weight: 700; cursor: pointer; background: transparent; color: #94a3b8; }
    .mode-btn.active { background: #0284c7; color: #ffffff; }
    .row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .legend { height: 10px; border-radius: 5px; background: linear-gradient(to right, #1d4ed8, #10b981, #fef08a, #f97316, #b91c1c); margin-top: 6px; }
    .l-txt { display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; margin-top: 3px; }
    .r-head { padding: 10px 18px 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94a3b8; }
    #list { flex: 1; overflow-y: auto; padding: 0 14px 14px; }
    .card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 8px; cursor: pointer; }
    .card:hover { border-color: #38bdf8; }
    .c-top { display: flex; justify-content: space-between; align-items: center; font-weight: 700; font-size: 13px; }
    .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; }
    .c-sub { font-size: 11px; color: #94a3b8; margin-top: 4px; display: flex; justify-content: space-between; }
    #map { flex: 1; height: 100%; }
    .leaflet-popup-content-wrapper, .leaflet-popup-tip { background: #1e293b; color: #f8fafc; }
    table { width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 6px; }
    td { padding: 3px 0; border-bottom: 1px solid #334155; }
    td:last-child { text-align: right; font-weight: 600; }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="header">
      <h1>Anomalía Hídrica SEAS5 (6 Meses)</h1>
      <p>Pronóstico vs. Climatología Histórica (Hindcast)</p>
      <span class="meta-tag">Corrida SEAS5: __FECHA_EMISION__ • Act: __ACTUALIZADO__</span>
    </div>
    <div class="controls">
      <div class="mode-box">
        <button class="mode-btn active" id="btn-clima" onclick="cambiarModo('clima')">Amenaza Climática<br><small style="font-weight:400;">(Escala idéntica al mapa)</small></button>
        <button class="mode-btn" id="btn-oper" onclick="cambiarModo('operativo')">Riesgo Operativo<br><small style="font-weight:400;">(Clima + Nivel Embalse)</small></button>
      </div>
      <div class="row"><label><input type="checkbox" id="chk-raster" checked /> Capa Anomalía Precipitación (%)</label></div>
      <div class="row" style="font-size:12px;"><span>Opacidad raster:</span><input type="range" id="rng-op" min="0" max="1" step="0.05" value="0.76" /></div>
      <div class="legend"></div>
      <div class="l-txt"><span>+35% (Húmedo)</span><span>0% (Normal)</span><span>-35% (Sequía)</span></div>
      <div class="row" style="margin-top:10px;"><label><input type="checkbox" id="chk-cuencas" /> Ver todas las cuencas aguas arriba</label></div>
    </div>
    <div class="r-head" id="titulo-ranking">Ranking por Amenaza Climática en Cuenca</div>
    <div id="list"></div>
  </div>
  <div id="map"></div>
  <script>
    const REPRESAS = __REPRESAS_JSON__;
    const CUENCAS = __CUENCAS_JSON__;
    const BOUNDS = __BOUNDS_JSON__;
    const IMG_B64 = "data:image/png;base64,__IMG_B64__";
    let modoActual = "clima";

    const map = L.map("map").setView([-18.5, -58.0], 4);
    const dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png").addTo(map);
    const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}");
    L.control.layers({ "Oscuro": dark, "Satélite": sat }).addTo(map);

    const raster = L.imageOverlay(IMG_B64, BOUNDS, { opacity: 0.76, interactive: false }).addTo(map);
    document.getElementById("chk-raster").onchange = e => e.target.checked ? raster.addTo(map) : map.removeLayer(raster);
    document.getElementById("rng-op").oninput = e => raster.setOpacity(parseFloat(e.target.value));

    const cTodas = L.geoJSON(CUENCAS, { style: { color: "#38bdf8", weight: 1, fillOpacity: 0.08 } });
    const cActiva = L.geoJSON(null, { style: { color: "#facc15", weight: 2.5, fillColor: "#fef08a", fillOpacity: 0.25 } }).addTo(map);
    document.getElementById("chk-cuencas").onchange = e => e.target.checked ? cTodas.addTo(map) : map.removeLayer(cTodas);

    function verCuenca(nom) {
      cActiva.clearLayers();
      const m = CUENCAS.features.filter(f => f.properties && f.properties.nombre === nom);
      if (m.length) cActiva.addData({ type: "FeatureCollection", features: m });
    }

    function colorEscalaUnificada(score) {
      if (score >= 72) return { bg: "#b91c1c", txt: "#fff", lbl: "Sequía Severa" };
      if (score >= 56) return { bg: "#f97316", txt: "#fff", lbl: "Déficit Moderado" };
      if (score >= 42) return { bg: "#eab308", txt: "#0f172a", lbl: "Cercano a Normal" };
      if (score >= 25) return { bg: "#10b981", txt: "#0f172a", lbl: "Ligeramente Húmedo" };
      return { bg: "#1d4ed8", txt: "#fff", lbl: "Superávit Hídrico" };
    }

    function cambiarModo(nuevoModo) {
      modoActual = nuevoModo;
      document.getElementById("btn-clima").classList.toggle("active", nuevoModo === "clima");
      document.getElementById("btn-oper").classList.toggle("active", nuevoModo === "operativo");
      document.getElementById("titulo-ranking").textContent = nuevoModo === "clima"
        ? "Ranking por Amenaza Climática en Cuenca"
        : "Ranking por Riesgo Operativo (Clima + Embalse)";
      render();
    }

    const markers = L.layerGroup().addTo(map);
    function render() {
      markers.clearLayers();
      const list = document.getElementById("list");
      list.innerHTML = "";
      const campoScore = modoActual === "clima" ? "riesgo_climatico" : "riesgo_operativo";
      const ordenadas = [...REPRESAS.features].sort((a, b) => (b.properties[campoScore] || 0) - (a.properties[campoScore] || 0));

      ordenadas.forEach((f, i) => {
        const p = f.properties;
        const [lon, lat] = f.geometry.coordinates;
        const score = p[campoScore] || 0;
        const c = colorEscalaUnificada(score);
        const signo = p.anomalia_pct > 0 ? "+" : "";
        const etiquetaBadge = modoActual === "clima" ? `${signo}${p.anomalia_pct}%` : `${score}%`;

        const mk = L.circleMarker([lat, lon], {
          radius: Math.max(7, Math.min(15, 6 + (p.capacidad_mw || 1000) / 1400)),
          fillColor: c.bg, color: "#fff", weight: 1.8, fillOpacity: 0.95
        }).bindPopup(`
          <b>#${i + 1} ${p.nombre}</b><br><small>${p.pais} • ${p.cuenca}</small>
          <table>
            <tr><td>Anomalía Cuenca (6m):</td><td style="color:${c.bg}">${signo}${p.anomalia_pct}%</td></tr>
            <tr><td>Índice Amenaza Clima:</td><td>${p.riesgo_climatico}%</td></tr>
            <tr><td>Nivel Actual Embalse:</td><td>${p.nivel_embalse}%</td></tr>
            <tr><td>Riesgo Operativo Total:</td><td><b>${p.riesgo_operativo}%</b></td></tr>
            <tr><td>Capacidad / Subcuencas:</td><td>${p.capacidad_mw.toLocaleString()} MW / ${p.subcuencas_aporte}</td></tr>
          </table>
        `);
        mk.on("click", () => verCuenca(p.nombre));
        mk.addTo(markers);

        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<div class="c-top"><span>${i + 1}. ${p.nombre}</span><span class="badge" style="background:${c.bg};color:${c.txt}">${etiquetaBadge}</span></div>
          <div class="c-sub"><span>${p.pais} • Anom: <b>${signo}${p.anomalia_pct}%</b></span><span>Embalse: <b>${p.nivel_embalse}%</b></span></div>`;
        card.onclick = () => { map.setView([lat, lon], 6); mk.openPopup(); verCuenca(p.nombre); };
        list.appendChild(card);
      });
    }
    render();
  </script>
</body>
</html>"""

html_final = (
    html_template
    .replace("__FECHA_EMISION__", fecha_emision_str)
    .replace("__ACTUALIZADO__", actualizado_utc)
    .replace("__REPRESAS_JSON__", represas.to_json())
    .replace("__CUENCAS_JSON__", cuencas_export.to_json())
    .replace("__BOUNDS_JSON__", json.dumps(bounds_leaflet))
    .replace("__IMG_B64__", png_b64)
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("index.html generado correctamente.")
