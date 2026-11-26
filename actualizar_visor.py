import os
import zipfile
import json
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
# 1. CENTRALES CON LOS MISMOS IDs DEL VISOR 3D
# ==========================================
centrales_data = [
    {"id": "ITAIPU",            "nombre": "Itaipú",            "pais": "Brasil / Paraguay",    "cuenca": "Río Paraná",        "capacidad_mw": 14000, "nivel_embalse": 92, "lon": -54.588, "lat": -25.408},
    {"id": "BELO_MONTE",        "nombre": "Belo Monte",        "pais": "Brasil",               "cuenca": "Río Xingu",         "capacidad_mw": 11233, "nivel_embalse": 74, "lon": -51.777, "lat": -3.123},
    {"id": "GURI",              "nombre": "Guri",              "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 10235, "nivel_embalse": 84, "lon": -62.998, "lat": 7.766},
    {"id": "TUCURUI",           "nombre": "Tucuruí",           "pais": "Brasil",               "cuenca": "Río Tocantins",     "capacidad_mw": 8370,  "nivel_embalse": 79, "lon": -49.648, "lat": -3.831},
    {"id": "PAULO_AFONSO",      "nombre": "Paulo Afonso",      "pais": "Brasil",               "cuenca": "Río São Francisco", "capacidad_mw": 4279,  "nivel_embalse": 58, "lon": -38.214, "lat": -9.397},
    {"id": "JIRAU",             "nombre": "Jirau",             "pais": "Brasil",               "cuenca": "Río Madeira",       "capacidad_mw": 3750,  "nivel_embalse": 71, "lon": -64.643, "lat": -9.266},
    {"id": "SANTO_ANTONIO",     "nombre": "Santo Antônio",     "pais": "Brasil",               "cuenca": "Río Madeira",       "capacidad_mw": 3568,  "nivel_embalse": 72, "lon": -63.951, "lat": -8.802},
    {"id": "ILHA_SOLTEIRA",     "nombre": "Ilha Solteira",     "pais": "Brasil",               "cuenca": "Río Paraná",        "capacidad_mw": 3444,  "nivel_embalse": 77, "lon": -51.365, "lat": -20.382},
    {"id": "YACYRETA",          "nombre": "Yacyretá",          "pais": "Argentina / Paraguay", "cuenca": "Río Paraná",        "capacidad_mw": 3200,  "nivel_embalse": 91, "lon": -56.737, "lat": -27.482},
    {"id": "MACAGUA",           "nombre": "Macagua",           "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 3167,  "nivel_embalse": 85, "lon": -62.668, "lat": 8.304},
    {"id": "XINGO",             "nombre": "Xingó",             "pais": "Brasil",               "cuenca": "Río São Francisco", "capacidad_mw": 3162,  "nivel_embalse": 61, "lon": -37.965, "lat": -9.625},
    {"id": "HIDROITUANGO",      "nombre": "Hidroituango",      "pais": "Colombia",             "cuenca": "Río Cauca",         "capacidad_mw": 2400,  "nivel_embalse": 82, "lon": -75.664, "lat": 7.135},
    {"id": "CARUACHI",          "nombre": "Caruachi",          "pais": "Venezuela",            "cuenca": "Río Caroní",        "capacidad_mw": 2160,  "nivel_embalse": 83, "lon": -62.798, "lat": 8.157},
    {"id": "SALTO_GRANDE",      "nombre": "Salto Grande",      "pais": "Argentina / Uruguay",  "cuenca": "Río Uruguay",       "capacidad_mw": 1890,  "nivel_embalse": 76, "lon": -57.938, "lat": -31.275},
    {"id": "PIEDRA_DEL_AGUILA", "nombre": "Piedra del Águila", "pais": "Argentina",            "cuenca": "Río Limay",         "capacidad_mw": 1400,  "nivel_embalse": 64, "lon": -70.013, "lat": -40.078},
    {"id": "GUAVIO",            "nombre": "Del Guavio",        "pais": "Colombia",             "cuenca": "Río Guavio",        "capacidad_mw": 1213,  "nivel_embalse": 69, "lon": -73.508, "lat": 4.723},
    {"id": "EL_CHOCON",         "nombre": "El Chocón",         "pais": "Argentina",            "cuenca": "Río Limay",         "capacidad_mw": 1200,  "nivel_embalse": 23, "lon": -68.751, "lat": -39.273},
    {"id": "MANTARO",           "nombre": "Mantaro",           "pais": "Perú",                 "cuenca": "Río Mantaro",       "capacidad_mw": 1008,  "nivel_embalse": 53, "lon": -74.654, "lat": -12.213},
    {"id": "RALCO",             "nombre": "Ralco",             "pais": "Chile",                "cuenca": "Río Biobío",        "capacidad_mw": 690,   "nivel_embalse": 21, "lon": -71.552, "lat": -37.998},
    {"id": "PENOL_GUATAPE",     "nombre": "Guatapé",           "pais": "Colombia",             "cuenca": "Río Nare",          "capacidad_mw": 560,   "nivel_embalse": 67, "lon": -75.161, "lat": 6.232}
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
# 3. CUENCAS AGUAS ARRIBA (HYDROBASINS LOCAL)
# ==========================================
print("Procesando cuencas HydroBASINS...")
shp_cuencas = "hybas_sa_lev06_v1c.shp"
if not os.path.exists(shp_cuencas):
    with zipfile.ZipFile("hybas_sa_lev06_v1c.zip", "r") as z:
        z.extractall(".")

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
resultados_represas = {}
for idx, s in enumerate(stats):
    anom = round(float(s["mean"]), 1) if s["mean"] is not None else 0.0
    r_clima = round(float(np.clip(((LIMITE - anom) / (2.0 * LIMITE)) * 100.0, 0, 100)), 1)
    vuln_embalse = 100.0 - float(represas.loc[idx, "nivel_embalse"])
    r_oper = round((0.65 * r_clima) + (0.35 * vuln_embalse), 1)
    dam_id = represas.loc[idx, "id"]
    resultados_represas[dam_id] = {
        "anomalia_pct": anom,
        "riesgo_climatico": r_clima,
        "riesgo_operativo": r_oper,
        "subcuencas_aporte": int(num_sub[idx])
    }

# ==========================================
# 4. EXPORTAR PNG, JSON DE DATOS Y GEOJSON DE CUENCAS
# ==========================================
da_r = rioxarray.open_rasterio("anomalia_seas5_6m_sa.tif").squeeze()
min_lon, min_lat, max_lon, max_lat = da_r.rio.bounds()

vals = da_r.values.astype(float)
norm = np.clip((LIMITE - vals) / (2.0 * LIMITE), 0.0, 1.0)
cmap = LinearSegmentedColormap.from_list("anom_sa", ["#1d4ed8", "#10b981", "#fef08a", "#f97316", "#b91c1c"], N=256)
rgba = cmap(norm)
rgba[..., 3] = np.where(np.isnan(vals), 0.0, 1.0)
plt.imsave("sequia_anomalia_6m.png", rgba)

salida_json = {
    "fecha_emision": fecha_emision_str,
    "actualizado_utc": datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
    "bounds": {
        "west": float(min_lon),
        "south": float(min_lat),
        "east": float(max_lon),
        "north": float(max_lat)
    },
    "represas": resultados_represas
}

with open("datos_sequia_seas5.json", "w", encoding="utf-8") as f:
    json.dump(salida_json, f, ensure_ascii=False, indent=2)

cuencas_export = gdf_captacion.copy()
cuencas_export["id"] = represas["id"].values
cuencas_export["nombre"] = represas["nombre"].values
cuencas_export["geometry"] = cuencas_export.geometry.simplify(0.02, preserve_topology=True)
cuencas_export.to_file("cuencas_captacion.geojson", driver="GeoJSON")

print("Exportados sequia_anomalia_6m.png, datos_sequia_seas5.json y cuencas_captacion.geojson correctamente.")
