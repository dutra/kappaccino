#!/home/dutra/.conda/envs/lenstronomy/bin/python
"""The imaging and truth maps the figures need, downloaded and kept as plain files in one tree, data/imaging/<cluster>/<source>/.

    download_imaging.py download [--clusters hera ares hermes hephaestus m0416 as1063] [--kinds jwst ffsims ffsims_site hst]
                                 [--hst-scale 60|30] [--local-source DIR ...] [--dry-run]
    download_imaging.py zip [--out FILE] [--kinds jwst ffsims ffsims_site] [--with-catalogs]
    download_imaging.py unpack ARCHIVE [--force]
    download_imaging.py list

Layout (every entry is a real file; the script never creates symbolic links):

    <cluster>/ffsims/     one folder per FF-SIMS cluster (hera, ares, hermes, hephaestus) with every file its page on the
                          site served: simulation_hst_<band>.fits (seven HFF bands), simulation_subaru_R.fits, the
                          catalogs and region files and, for Hera and Ares, the lens results (kappa / gammax / gammay at
                          z = 9, deflAngles*.fits, <Cluster>_F.fits, <Cluster>_G.fits).  run.xsh reads the three ACS
                          bands and the three z = 9 maps.
    m0416/canucs_dr1/     CANUCS DR1 NIRCam mosaics (MAST HLSP): 20 mas F115W F150W F200W, 40 mas F444W, kept
                          uncompressed (.fits): the archive's .fits.gz is unpacked once, since reading cutouts through a
                          gzip stream is slow
    as1063/dja_v7.5/      DAWN JWST Archive Grizli v7.5 mosaics, the same bands
    <cluster>/buffalo/<scale>mas/   BUFFALO v1.0 HST mosaics of M0416 and AS1063 in the seven HFF bands (on request)
    ffsims_site/          the pages, stylesheet and thumbnails of the FF-SIMS site, for the record

Sources.  The FF-SIMS site (pico.oabo.inaf.it/~massimo/Public/FF, Meneghetti et al. 2017) has been offline since 2026
and the Wayback Machine holds only its text files, so the FITS come from a local copy given with --local-source (a
synced folder, an unpacked archive of this script, a wget mirror of the site); the original URLs stay in the manifest
and are tried first.  Files found under a --local-source root are copied (never linked), validated (gzip CRC, FITS
structure, celestial WCS for the mosaics) and given a .json sidecar with origin and sha256; data/imaging/manifest.json
lists the tree.  A symbolic link left by an earlier version is replaced by a copy of its target, or fetched again when
the target is gone.  `zip` packs the JWST files, the FF-SIMS files and the site pages (no HST; with the catalogs/ tree by
default) into one archive that `unpack` restores.  None of this is tracked by git (data/* and every FITS are ignored).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "imaging"
JWST_BANDS = ("F115W", "F150W", "F200W", "F444W")
HFF_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W")
FF_BANDS = ("F435W", "F606W", "F814W")
FF_TRUTH = ("kappa_z9_0.fits", "gammax_z9_0.fits", "gammay_z9_0.fits")
FF_CLUSTERS = ("hera", "ares", "hermes", "hephaestus")
CANUCS = "https://archive.stsci.edu/hlsps/canucs/dr1/macs0416/clu/"
DJA = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/"
BUFFALO = "https://archive.stsci.edu/hlsps/buffalo/{cluster}/images/{scale}mas/"
FFSIMS = "http://pico.oabo.inaf.it/~massimo/Public/FF/"
WAYBACK = "http://web.archive.org/web/2025id_/"
FF_RELEASE = "FF-SIMS (Meneghetti et al. 2017, MNRAS 472, 3177)"
# Every file the FF-SIMS site served per cluster, as <site folder>/<name>; the local copy flattens them into
# <cluster>/ffsims/<name> (no name is shared between the fits/ and results/ folders of one cluster).  Hera and Ares have
# the lens results; Hermes and Hephaestus only the images and catalogs.  finalshearHST.cat (Hera, Ares) and nome.txt /
# nome.reg (Hermes, Hephaestus) were linked by the site but never served, as far as the Wayback Machine knows.
FF_SITE_FILES = {
    "hera": ["fits/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshearHST.cat", "finalshear_z.cat", "galcat2.txt",
                                   "multimages.reg", "multimages.txt", "simulation_subaru_R.fits")]
            + [f"fits/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS]
            + ["results/" + n for n in ("deflAnglesHera.fits", "Hera_F.fits", "Hera_G.fits") + FF_TRUTH],
    "ares": ["fits/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "finalshearHST.cat",
                                   "multimages.reg", "multimages.txt", "simulation_subaru_R.fits")]
            + [f"fits/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS]
            + ["results/" + n for n in ("deflAnglesAres.fits", "Ares_F.fits", "Ares_G.fits") + FF_TRUTH],
    "hermes": ["fits/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "nome.reg", "nome.txt", "simulation_subaru_R.fits")]
              + [f"fits/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS],
    "hephaestus": ["fits/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "nome.reg", "nome.txt", "simulation_subaru_R.fits")]
                  + [f"fits/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS],
}
FF_SITE_PAGES = ["index.html", "instructions.html", "contacts.html", "hst_image.html", "subaru_image.html", "multiple_images.html",
                 "cluster_galaxies.html", "ares.html", "hera.html", "hermes.html", "hephaestus.html", "style.css",
                 "images/ares_thumb.jpg", "images/hera_thumb.jpg", "images/hermes_thumb.jpg", "images/hephaestus_thumb.jpg",
                 "images/udf_banner2.jpeg", "images/img03.png", "images/img04.jpg"]
ALL_CLUSTERS = FF_CLUSTERS + ("m0416", "as1063")


def ff_use(name: str) -> str:
    if name in FF_TRUTH:
        return "truth"
    if name in {f"simulation_hst_{b.lower()}.fits" for b in FF_BANDS}:
        return "rgb"
    return "archive"


def manifest(hst_scale: int = 60) -> list[dict]:
    """Every product: where it goes (relpath under data/imaging), where it comes from, what it is for."""
    out = []
    for band in JWST_BANDS:
        scale = 40 if band == "F444W" else 20
        name = f"hlsp_canucs_jwst_nircam_macs0416-clu-{scale}mas_{band.lower()}_v1_sci.fits"
        out.append(dict(cluster="m0416", source="canucs_dr1", kind="jwst", band=band, relpath=f"m0416/canucs_dr1/{name}",
                        urls=[CANUCS + name + ".gz"], archive_name=name + ".gz",
                        release="CANUCS DR1 (Rihtarsic et al. 2025); DOI 10.17909/18nv-np70", use="rgb"))
        name = f"abells1063-grizli-v7.5-{band.lower()}-clear_drc_sci.fits.gz"
        out.append(dict(cluster="as1063", source="dja_v7.5", kind="jwst", band=band, relpath=f"as1063/dja_v7.5/{name}",
                        urls=[DJA + name], release="DAWN JWST Archive, Grizli v7.5 mosaics", use="rgb"))
    for cluster, token in (("m0416", "macs0416"), ("as1063", "abells1063")):
        for band in HFF_BANDS:
            camera = "acs-wfc" if band in FF_BANDS else "wfc3-ir"
            name = f"hlsp_buffalo_hst_{camera}-{hst_scale}mas_{token}_{band.lower()}_v1.0_drz.fits"
            out.append(dict(cluster=cluster, source="buffalo", kind="hst", band=band, relpath=f"{cluster}/buffalo/{hst_scale}mas/{name}",
                            urls=[BUFFALO.format(cluster=token, scale=hst_scale) + name],
                            release="BUFFALO v1.0 (Steinhardt et al. 2020); DOI 10.17909/t9-k61t-qc38", use="rgb"))
    for cluster, paths in FF_SITE_FILES.items():
        for path in paths:
            folder, name = path.split("/", 1)
            band = name[len("simulation_hst_"):-len(".fits")].upper() if name.startswith("simulation_hst_") else None
            out.append(dict(cluster=cluster, source="ffsims", kind="ffsims", band=band, relpath=f"{cluster}/ffsims/{name}",
                            urls=[FFSIMS + f"{folder}/{cluster}/{name}", WAYBACK + FFSIMS + f"{folder}/{cluster}/{name}"],
                            release=FF_RELEASE, use=ff_use(name)))
    for path in FF_SITE_PAGES:
        out.append(dict(cluster="ffsims", source="ffsims_site", kind="ffsims_site", band=None, relpath=f"ffsims_site/{path}",
                        urls=[FFSIMS + path, WAYBACK + FFSIMS + path], release=FF_RELEASE + ", site pages", use="archive"))
    return out


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path: Path, product: dict) -> dict:
    """gzip CRC, FITS structure, a 2-d image with a celestial WCS for the mosaics (the FF-SIMS maps have a plain
    linear WCS); returns the metadata of the sidecar."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    is_fits = path.name.endswith((".fits", ".fits.gz"))
    if product["use"] == "archive":                       # kept for the record: a FITS must open, the rest is taken as is
        if is_fits:
            with fits.open(path, memmap=False) as hdus:
                hdus.verify("exception")
                shape = [int(n) for n in hdus[0].shape] if hdus[0].data is not None else []
        else:
            shape = []
        return dict(shape=shape, bytes=path.stat().st_size, sha256=sha256(path))
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 << 20):
                pass
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        hdu = next(h for h in hdus if h.header.get("NAXIS") == 2)
        header = hdu.header
        wcs = WCS(header)
        if product["kind"] != "ffsims" and not wcs.has_celestial:
            raise ValueError(f"{path}: no celestial WCS")
        scales = np.asarray(proj_plane_pixel_scales(wcs)) * (3600.0 if product["kind"] != "ffsims" or wcs.has_celestial else 1.0)
        meta = dict(shape=[int(header["NAXIS2"]), int(header["NAXIS1"])], pixel_scale_arcsec=[float(s) for s in scales],
                    bunit=header.get("BUNIT"), fits_filter=header.get("FILTER"))
    meta.update(bytes=path.stat().st_size, sha256=sha256(path))
    return meta


def curl(url: str, partial: Path) -> bool:
    result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3", "--retry-delay", "3",
                             "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-",
                             "--output", str(partial), url])
    return result.returncode == 0


def find_local(product: dict, roots: list[Path]) -> Path | None:
    """A regular file of the same name under any of the roots (synced copies and mirrors keep the archive names); for a
    product kept uncompressed, its plain name first, then the archive's gzipped name."""
    names = [Path(product["relpath"]).name] + ([product["archive_name"]] if product.get("archive_name") else [])
    cluster = product.get("cluster")
    for root in roots:
        if not root.exists():
            continue
        found = [c for name in names for c in sorted(root.rglob(name)) if c.is_file() and not c.is_symlink()]
        # FF-SIMS names repeat across clusters (kappa_z9_0.fits, clgal_cat.txt): the copy must sit under a folder named for it.
        if product["kind"] == "ffsims":
            found = [c for c in found if cluster in {part.lower() for part in c.relative_to(root).parts[:-1]}]
        if found:
            return found[0]
    return None


def materialize(destination: Path) -> str | None:
    """Replace a symbolic link by a copy of its target; returns the target, or None when it is gone (the link is removed)."""
    link = destination.readlink()
    target = link if link.is_absolute() else destination.parent / link
    if not target.exists():
        destination.unlink()
        print(f"dangling  {destination.relative_to(ROOT)} -> {link}: removed, fetching again")
        return None
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(target.resolve(), temporary)
    destination.unlink()
    temporary.replace(destination)
    print(f"copied    {destination.relative_to(ROOT)}  <-  {target.resolve()} (was a link)")
    return str(target.resolve())


def fetch(product: dict, *, local: list[Path], dry_run: bool) -> dict | None:
    destination = ROOT / product["relpath"]
    sidecar = destination.with_name(destination.name + ".json")
    origin = None
    if destination.is_symlink():
        if dry_run:
            print(f"link      {product['relpath']} -> {destination.readlink()}: would be replaced by a copy")
            return None
        target = materialize(destination)
        if target is not None:
            origin = {"origin": target}
    if destination.exists():
        if sidecar.exists() and origin is None:
            print(f"present   {product['relpath']}")
            return json.loads(sidecar.read_text())
        print(f"validate  {product['relpath']}")
        meta, origin = validate(destination, product), origin or {"origin": "present"}
    else:
        source = find_local(product, local)
        if dry_run:
            print(f"{'copy' if source else 'download':9s} {product['relpath']}  <-  {source or product['urls'][0]}")
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        if source is not None:
            print(f"copy      {product['relpath']}  <-  {source}")
            shutil.copy2(source, partial)
            origin = {"origin": str(source.resolve())}
        else:
            for url in product["urls"]:
                print(f"download  {product['relpath']}  <-  {url}", flush=True)
                if "web.archive.org" in url:
                    time.sleep(2.0)                               # the Wayback Machine rate-limits bursts (HTTP 429)
                if curl(url, partial):
                    origin = {"origin": url}
                    break
                partial.unlink(missing_ok=True)
            else:
                print(f"MISSING   {product['relpath']}: no source answered (give --local-source)")
                return None
        if product.get("archive_name") and _is_gzip(partial):
            gunzip(partial, destination)
        else:
            partial.replace(destination)
        meta = validate(destination, product)
    record = dict(product, **meta, **origin, path=str(destination), verified_utc=datetime.now(timezone.utc).isoformat())
    sidecar.write_text(json.dumps(record, indent=1) + "\n")
    return record


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(2) == b"\x1f\x8b"


def gunzip(source: Path, destination: Path) -> None:
    """Unpack ``source`` (gzip CRC checked to the end) to ``destination`` through a temporary file, then remove ``source``."""
    print(f"unpack    {destination.relative_to(ROOT)}", flush=True)
    temporary = destination.with_name(destination.name + ".unpacking")
    with gzip.open(source, "rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 << 20)
    temporary.replace(destination)
    source.unlink()


def write_manifest(records: list[dict]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                                                       files=records), indent=1) + "\n")
    (ROOT / "README.md").write_text(README)


def selected(kinds, clusters, hst_scale: int = 60) -> list[dict]:
    """--clusters narrows the data; the site pages always come whole."""
    return [p for p in manifest(hst_scale) if p["kind"] in kinds and (p["cluster"] in clusters or p["kind"] == "ffsims_site")]


def cmd_download(args) -> int:
    local = [Path(p) for p in (args.local_source or [])]
    records, missing = [], 0
    for product in selected(args.kinds, args.clusters, args.hst_scale):
        record = fetch(product, local=local, dry_run=args.dry_run)
        if record is None and not args.dry_run:
            missing += 1
        elif record is not None:
            records.append(record)
    if not args.dry_run:
        present = [json.loads(p.read_text()) for p in sorted(ROOT.rglob("*.json")) if p.name != "manifest.json"]
        write_manifest(present)
        links = [p for p in ROOT.rglob("*") if p.is_symlink()]
        if links:
            print(f"WARNING: {len(links)} symbolic links remain under {ROOT} (run download for their kinds to replace them)")
        print(f"{len(records)} ready, {missing} missing; manifest {ROOT / 'manifest.json'}")
    return 1 if missing else 0


def cmd_zip(args) -> int:
    out = Path(args.out)
    members: list[tuple[Path, str]] = []
    missing = 0
    for product in selected(args.kinds, args.clusters):
        path = ROOT / product["relpath"]
        if not path.exists():
            if product["use"] == "archive":
                print(f"missing   {product['relpath']} (left out)")
                missing += 1
                continue
            print(f"MISSING   {product['relpath']} (run download first)")
            return 1
        members.append((path, f"imaging/{product['relpath']}"))
        sidecar = path.with_name(path.name + ".json")
        if sidecar.exists():
            members.append((sidecar, f"imaging/{product['relpath']}.json"))
    for name in ("README.md", "manifest.json"):
        if (ROOT / name).exists():
            members.append((ROOT / name, f"imaging/{name}"))
    if args.with_catalogs:
        for path in sorted((REPO / "catalogs").rglob("*")):
            if path.is_file() and not path.name.endswith((".fits", ".fits.gz")):
                members.append((path, f"catalogs/{path.relative_to(REPO / 'catalogs')}"))
    total = sum(p.stat().st_size for p, _ in members)
    print(f"{len(members)} files, {total / 1e9:.2f} GB -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", allowZip64=True) as archive:
        for path, arcname in members:
            # gzipped mosaics do not shrink again; the plain FITS of FF-SIMS and the text files do.
            method = zipfile.ZIP_STORED if path.name.endswith((".gz", ".jpg", ".jpeg")) else zipfile.ZIP_DEFLATED
            archive.write(path, arcname, compress_type=method)
            print(f"  {arcname}", flush=True)
    print(f"written {out} ({out.stat().st_size / 1e9:.2f} GB); {missing} archive files were not available")
    return 0


def cmd_unpack(args) -> int:
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        for name in names:
            target = (REPO / "data" / name) if name.startswith("imaging/") else (REPO / name)
            if target.exists() and not args.force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            print(f"  {target.relative_to(REPO)}")
    print(f"unpacked {len(names)} entries under {ROOT} (and catalogs/)")
    return 0


def cmd_list(args) -> int:
    for product in manifest(args.hst_scale):
        path = ROOT / product["relpath"]
        state = "LINK" if path.is_symlink() else "file" if path.exists() else "-"
        print(f"{state:5s} {product['kind']:11s} {product['use']:7s} {product['relpath']}")
    return 0


README = """# data/imaging

The mosaics and truth maps the figures read, plain files, one folder per cluster and source (scripts/download_imaging.py):

    hera/ffsims/      every FF-SIMS file of the cluster (Meneghetti et al. 2017): simulation_hst_<band>.fits (seven HFF bands),
                      simulation_subaru_R.fits, the catalogs and region files, and the lens results kappa_z9_0 / gammax_z9_0 /
                      gammay_z9_0.fits, deflAnglesHera.fits, Hera_F.fits, Hera_G.fits.  run.xsh reads the three ACS bands
                      and the three z = 9 maps.
    ares/ffsims/      the same for Ares
    hermes/ffsims/, hephaestus/ffsims/   images and catalogs only (the site published no lens results for them)
    m0416/canucs_dr1/ hlsp_canucs_jwst_nircam_macs0416-clu-{20,40}mas_<band>_v1_sci.fits, F115W F150W F200W F444W (CANUCS DR1,
                      unpacked from the archive's .fits.gz)
    m0416/buffalo/    <scale>mas/hlsp_buffalo_hst_*_macs0416_<band>_v1.0_drz.fits, the seven HFF bands (HST, BUFFALO v1.0)
    as1063/dja_v7.5/  abells1063-grizli-v7.5-<band>-clear_drc_sci.fits.gz, F115W F150W F200W F444W (DAWN JWST Archive)
    as1063/buffalo/   as for m0416
    ffsims_site/      the pages, stylesheet and thumbnails of the FF-SIMS site, which went offline in 2026

Each file has a .json sidecar (origin URL or local path, sha256, shape, pixel scale) and manifest.json lists them.
run.xsh points ImageCatalogCutoutConfig.image_dir at <cluster>/<source> and the truth maps at <cluster>/ffsims.
Nothing here is a symbolic link: `download --local-source DIR` copies files it finds under DIR.  `zip` packs the JWST
and FF-SIMS files with the catalogs/ tree; `unpack` restores the archive.  None of it is tracked by git.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    clusters = dict(nargs="+", choices=ALL_CLUSTERS, default=list(ALL_CLUSTERS))
    kinds = dict(nargs="+", choices=("jwst", "ffsims", "ffsims_site", "hst"), default=["jwst", "ffsims", "ffsims_site"])
    d = sub.add_parser("download", help="fetch or copy the files into data/imaging")
    d.add_argument("--clusters", **clusters)
    d.add_argument("--kinds", **kinds, help="jwst, ffsims and the site pages by default; hst adds the BUFFALO mosaics (0.7 GB per band at 60 mas)")
    d.add_argument("--hst-scale", type=int, choices=(60, 30), default=60)
    d.add_argument("--local-source", nargs="+", help="roots searched for copies of the same file names (a synced folder, an unpacked archive, a site mirror)")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(run=cmd_download)
    z = sub.add_parser("zip", help="pack the JWST and FF-SIMS files (no HST) and the catalogs into one archive")
    z.add_argument("--out", default=str(REPO / "data" / "lenscluster_imaging_jwst_ffsims.zip"))
    z.add_argument("--clusters", **clusters)
    z.add_argument("--kinds", **kinds)
    z.add_argument("--with-catalogs", action=argparse.BooleanOptionalAction, default=True)
    z.set_defaults(run=cmd_zip)
    u = sub.add_parser("unpack", help="restore an archive made by zip into data/imaging and catalogs/")
    u.add_argument("archive")
    u.add_argument("--force", action="store_true", help="overwrite files that exist")
    u.set_defaults(run=cmd_unpack)
    ls = sub.add_parser("list", help="the manifest and what is present")
    ls.add_argument("--hst-scale", type=int, choices=(60, 30), default=60)
    ls.set_defaults(run=cmd_list)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
