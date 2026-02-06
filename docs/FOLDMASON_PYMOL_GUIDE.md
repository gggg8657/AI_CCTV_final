# FoldMason & PyMOL 종합 가이드

> 단백질 구조 정렬 및 시각화를 위한 도구 문서

---

## 목차

1. [FoldMason 개요](#1-foldmason-개요)
2. [FoldMason 설치](#2-foldmason-설치)
3. [FoldMason CLI 사용법](#3-foldmason-cli-사용법)
4. [FoldMason 주요 모듈](#4-foldmason-주요-모듈)
5. [FoldMason 파라미터](#5-foldmason-파라미터)
6. [관련 도구 비교](#6-관련-도구-비교-foldmason-vs-foldseek-vs-tm-align)
7. [PyMOL 개요](#7-pymol-개요)
8. [PyMOL 설치](#8-pymol-설치)
9. [PyMOL 기본 사용법](#9-pymol-기본-사용법)
10. [PyMOL Python API](#10-pymol-python-api)
11. [PyMOL 스크립팅 예제](#11-pymol-스크립팅-예제)
12. [FoldMason + PyMOL 통합 워크플로우](#12-foldmason--pymol-통합-워크플로우)
13. [참고 자료](#13-참고-자료)

---

## 1. FoldMason 개요

### 1.1 정의

**FoldMason**은 대규모 단백질 구조 세트의 **다중 구조 정렬(Multiple Structure Alignment, MSTA)**을 위한 도구입니다.

### 1.2 주요 특징

| 특징 | 설명 |
|------|------|
| **대규모 처리** | 수십만 개의 단백질 구조를 정렬 가능 |
| **고속 처리** | 기존 MSTA 방법보다 **2자릿수(100배) 빠름** |
| **정확도** | 최신 MSTA 방법과 동등하거나 더 높은 정렬 품질 |
| **3Di 알파벳** | Foldseek의 구조 알파벳 사용 |
| **신뢰도 점수** | 정렬에 대한 confidence score 계산 |
| **시각화** | 인터랙티브 HTML 시각화 제공 |
| **계통분석** | twilight zone 이하의 계통 분석 지원 |

### 1.3 활용 분야

- **원거리 관련 단백질 분석**: 서열 유사성이 낮지만 구조적 보존이 있는 단백질
- **계통 분석**: Flaviviridae glycoprotein 등의 진화 분석
- **대규모 구조 비교**: AlphaFold 예측 구조의 대규모 비교

### 1.4 논문

```
Gilchrist CLM, Mirdita M, and Steinegger M. 
Multiple Protein Structure Alignment at Scale with FoldMason. 
Science, Vol 391, Issue 6784 (2026)
bioRxiv, doi:10.1101/2024.08.01.606130 (2024)
```

---

## 2. FoldMason 설치

### 2.1 Pre-compiled Binary (권장)

```bash
# Linux AVX2 (가장 빠름, CPU 지원 확인: cat /proc/cpuinfo | grep avx2)
wget https://mmseqs.com/foldmason/foldmason-linux-avx2.tar.gz
tar xvzf foldmason-linux-avx2.tar.gz
export PATH=$(pwd)/foldmason/bin/:$PATH

# Linux SSE2 (구형 CPU)
wget https://mmseqs.com/foldmason/foldmason-linux-sse2.tar.gz
tar xvzf foldmason-linux-sse2.tar.gz
export PATH=$(pwd)/foldmason/bin/:$PATH

# Linux ARM64
wget https://mmseqs.com/foldmason/foldmason-linux-arm64.tar.gz
tar xvzf foldmason-linux-arm64.tar.gz
export PATH=$(pwd)/foldmason/bin/:$PATH

# macOS (Universal)
wget https://mmseqs.com/foldmason/foldmason-osx-universal.tar.gz
tar xvzf foldmason-osx-universal.tar.gz
export PATH=$(pwd)/foldmason/bin/:$PATH
```

### 2.2 Conda 설치

```bash
# Conda/Mamba 설치
conda install -c conda-forge -c bioconda foldmason

# 또는 새 환경 생성
mamba create -n foldmason foldmason
conda activate foldmason
```

### 2.3 Docker

```bash
docker pull quay.io/biocontainers/foldmason:<tag>
```

### 2.4 설치 확인

```bash
foldmason -h
```

출력 예시:
```
FoldMason is a progressive aligner for fast and accurate multiple alignment 
of hundreds of thousands of protein structures.
```

---

## 3. FoldMason CLI 사용법

### 3.1 기본 다중 정렬 (easy-msa)

```bash
# 기본 사용법
foldmason easy-msa <PDB/mmCIF files> result.fasta tmpFolder

# HTML 리포트 포함
foldmason easy-msa <PDB/mmCIF files> result.fasta tmpFolder --report-mode 1

# 예제 실행
foldmason easy-msa ./structures/*.pdb output.fasta tmp/ --report-mode 1
```

### 3.2 출력 파일

| 파일 | 설명 |
|------|------|
| `result_aa.fa` | 아미노산 서열 FASTA 정렬 |
| `result_3di.fa` | 3Di 구조 알파벳 FASTA 정렬 |
| `result.nw` | Newick 형식 계통수 |
| `result.html` | 인터랙티브 HTML 시각화 (--report-mode 1) |
| `result.json` | JSON 데이터 (--report-mode 2) |

### 3.3 대규모 데이터셋 정렬

```bash
# Foldseek 사전 클러스터링 사용 (대규모 구조 세트)
foldmason easy-msa <PDB/mmCIF files> result tmpFolder --precluster
```

### 3.4 단계별 실행

```bash
# 1. 데이터베이스 생성
foldmason createdb <PDB/mmCIF files> myDb

# 2. 구조 다중 정렬
foldmason structuremsa myDb result

# 3. LDDT 리포트 생성
foldmason msa2lddtreport myDb result_aa.fa result.html --guide-tree result.nw

# 4. (선택) MSA 개선
foldmason refinemsa myDb result_aa.fa refined_result.fa --refine-iters 3
```

---

## 4. FoldMason 주요 모듈

| 모듈 | 설명 | 사용법 |
|------|------|--------|
| `easy-msa` | 구조 파일에서 다중 정렬 워크플로우 | `foldmason easy-msa <files> result tmp/` |
| `structuremsa` | 구조 데이터베이스에서 다중 정렬 | `foldmason structuremsa myDb result` |
| `msa2lddt` | MSA의 구조 기반 점수(LDDT) 계산 | `foldmason msa2lddt myDb result.fa score` |
| `msa2lddtreport` | LDDT 계산 및 HTML 리포트 생성 | `foldmason msa2lddtreport myDb result.fa result.html` |
| `msa2lddtjson` | LDDT 계산 및 JSON 출력 | `foldmason msa2lddtjson myDb result.fa result.json` |
| `refinemsa` | 구조 정보를 사용한 반복적 MSA 개선 | `foldmason refinemsa myDb result.fa refined.fa` |
| `createdb` | 구조 데이터베이스 생성 | `foldmason createdb <files> structureDB` |
| `structuremsacluster` | 클러스터 DB에서 다중 정렬 | `foldmason structuremsacluster queryDB clusterDB result` |

---

## 5. FoldMason 파라미터

### 5.1 정렬 파라미터

| 옵션 | 카테고리 | 설명 | 기본값 |
|------|----------|------|--------|
| `--gap-open` | Alignment | Gap opening penalty | 10 |
| `--gap-extend` | Alignment | Gap extension penalty | 1 |
| `--refine-iters` | Alignment | 정렬 후 개선 반복 횟수 | 0 |
| `--output-mode` | Alignment | 0: 아미노산, 1: 3Di 알파벳 | 0 |
| `--match-ratio` | Alignment | 유지할 컬럼의 잔기 비율 | 0.9 |

### 5.2 점수 파라미터

| 옵션 | 카테고리 | 설명 | 기본값 |
|------|----------|------|--------|
| `--pair-threshold` | Scoring | LDDT 계산을 위한 갭 비율 임계값 | 0.0 |

### 5.3 리포트 파라미터

| 옵션 | 설명 |
|------|------|
| `--report-mode 0` | 리포트 없음 (기본) |
| `--report-mode 1` | HTML 리포트 생성 |
| `--report-mode 2` | JSON 데이터 생성 |

---

## 6. 관련 도구 비교: FoldMason vs Foldseek vs TM-align

| 도구 | 주요 기능 | 장점 | 한계 | 관계 |
|------|----------|------|------|------|
| **FoldMason** | 대규모 다중 구조 정렬 (MSTA) | 100,000+ 구조 처리; 3Di + TM-align; 신뢰도 점수; 계통 분석 | Pairwise aligner에 의존 | Foldseek + TM-align 활용 |
| **Foldseek** | 빠른 pairwise 구조 검색/클러스터링 | 초고속 (CPU/GPU); 3Di+AA 정렬; TM-score/LDDT 출력 | Pairwise만 가능 | FoldMason의 사전 정렬에 사용 |
| **TM-align** | Pairwise 구조 정렬 | 고정확도 global/local 정렬; TM-score 정규화 | 대규모에서 느림; Pairwise만 | Foldseek/FoldMason에 통합 |

### 6.1 Foldseek 기본 사용법

```bash
# Pairwise 구조 검색
foldseek easy-search query.pdb targetDB/ aln.tsv tmp/

# TM-align 모드 (global alignment)
foldseek easy-search query.pdb targetDB/ aln.tsv tmp/ --alignment-type 1

# 데이터베이스 생성 후 검색
foldseek createdb query/ queryDB
foldseek createdb target/ targetDB
foldseek search queryDB targetDB aln tmp/ -a
foldseek aln2tmscore queryDB targetDB aln aln_tmscore
foldseek createtsv queryDB targetDB aln_tmscore aln.tsv
```

---

## 7. PyMOL 개요

### 7.1 정의

**PyMOL**은 Python으로 작성된 강력한 **분자 시각화 시스템**으로, 구조 생물학, 약물 설계, 분자 모델링에서 고품질 3D 이미지와 애니메이션을 생성하는 데 널리 사용됩니다.

### 7.2 주요 특징

| 특징 | 설명 |
|------|------|
| **3D 시각화** | 단백질, DNA, 리간드, 복합체의 고품질 렌더링 |
| **다양한 표현** | Lines, Sticks, Spheres, Cartoon, Surface 등 |
| **Ray-tracing** | 출판 품질의 이미지 생성 |
| **Python 스크립팅** | 자동화 및 커스텀 워크플로우 |
| **측정 기능** | 결합 거리, 각도, 이면각 측정 |
| **다중 포맷** | PDB, MOL, MOL2, SDF, XYZ 등 지원 |

### 7.3 선택 모드

| 모드 | 설명 |
|------|------|
| Residues | 잔기 단위 선택 |
| Molecules | 분자 단위 선택 |
| Chains | 체인 단위 선택 |
| Objects | 객체 단위 선택 |
| C-alphas | Cα 원자만 선택 |
| Segments | 세그먼트 단위 선택 |

---

## 8. PyMOL 설치

### 8.1 공식 다운로드

- **오픈소스 빌드**: https://pymol.org
- **Schrödinger 버전** (무료 교육용): https://pymol.org/edu

### 8.2 패키지 매니저

```bash
# Ubuntu/Debian
sudo apt-get install pymol

# Conda
conda install -c conda-forge pymol-open-source

# macOS (Homebrew)
brew install pymol
```

### 8.3 설치 확인

```bash
pymol --version
# 또는 GUI 실행
pymol
```

---

## 9. PyMOL 기본 사용법

### 9.1 마우스 조작

| 동작 | 조작 |
|------|------|
| 회전 | 왼쪽 버튼 드래그 |
| 줌 | 스크롤 휠 또는 오른쪽 버튼 드래그 |
| 이동 | Ctrl + 클릭 + 드래그 |
| 선택 | 클릭 (핑크색 하이라이트) |

### 9.2 기본 명령어

```
# 구조 로드
fetch 1atp

# 표현 변경
show cartoon
show sticks, resn ATP
hide lines

# 색상 적용
color red, chain A
color bychain

# 선택
select active_site, resi 50-100 and chain A

# 이미지 저장
ray 1920, 1080
png my_image.png

# 뷰 저장/복원
get_view
set_view (...)
```

### 9.3 표현(Representation) 종류

| 표현 | 설명 | 명령 |
|------|------|------|
| Lines | 선 표현 | `show lines` |
| Sticks | 막대 표현 | `show sticks` |
| Spheres | 구 표현 (VDW) | `show spheres` |
| Cartoon | 2차 구조 (헬릭스, 시트) | `show cartoon` |
| Surface | 분자 표면 | `show surface` |
| Ribbon | 리본 표현 | `show ribbon` |
| Mesh | 메쉬 표면 | `show mesh` |

---

## 10. PyMOL Python API

### 10.1 기본 구조

```python
from pymol import cmd

# 또는
from pymol import *
```

### 10.2 주요 함수

| 함수 | 설명 | 예제 |
|------|------|------|
| `cmd.fetch(pdb_id, name)` | PDB 구조 다운로드 | `cmd.fetch("1atp", "kinase")` |
| `cmd.load(filename, name)` | 파일 로드 | `cmd.load("protein.pdb", "prot")` |
| `cmd.select(name, selection)` | 선택 생성 | `cmd.select("helix", "ss h")` |
| `cmd.show(rep, selection)` | 표현 표시 | `cmd.show("cartoon", "all")` |
| `cmd.hide(rep, selection)` | 표현 숨김 | `cmd.hide("lines", "all")` |
| `cmd.color(color, selection)` | 색상 적용 | `cmd.color("red", "chain A")` |
| `cmd.orient(selection)` | 뷰 자동 조정 | `cmd.orient("all")` |
| `cmd.ray(width, height)` | Ray-trace 이미지 | `cmd.ray(1920, 1080)` |
| `cmd.png(filename)` | PNG 저장 | `cmd.png("output.png")` |
| `cmd.save(filename, selection)` | 구조 저장 | `cmd.save("out.pdb", "all")` |
| `cmd.align(mobile, target)` | 구조 정렬 | `cmd.align("mobile", "target")` |
| `cmd.cealign(target, mobile)` | CE 정렬 | `cmd.cealign("target", "mobile")` |
| `cmd.scene(name, action)` | 씬 저장/복원 | `cmd.scene("scene1", "store")` |
| `cmd.get_view()` | 현재 뷰 가져오기 | `view = cmd.get_view()` |
| `cmd.set_view(view)` | 뷰 설정 | `cmd.set_view(view)` |

### 10.3 선택 문법

| 문법 | 설명 | 예제 |
|------|------|------|
| `chain X` | 특정 체인 | `chain A` |
| `resi N` | 잔기 번호 | `resi 50-100` |
| `resn XXX` | 잔기 이름 | `resn ALA` |
| `name X` | 원자 이름 | `name CA` |
| `elem X` | 원소 | `elem C` |
| `ss X` | 2차 구조 (h=helix, s=sheet) | `ss h` |
| `and`, `or`, `not` | 논리 연산 | `chain A and resi 50-100` |
| `within N of X` | 거리 내 선택 | `within 5 of resn ATP` |
| `byres X` | 잔기 단위 확장 | `byres within 5 of resn ATP` |

---

## 11. PyMOL 스크립팅 예제

### 11.1 기본 시각화 스크립트

```python
#!/usr/bin/env python
"""basic_visualization.py - 단백질 기본 시각화"""

from pymol import cmd

# 구조 로드
cmd.fetch("1atp", "kinase", async_=0)

# 표현 설정
cmd.show("cartoon", "all")
cmd.hide("lines", "all")

# 체인별 색상
cmd.color("cyan", "chain E")
cmd.color("orange", "chain I")

# ATP 리간드 강조
cmd.select("atp", "resn ATP")
cmd.show("sticks", "atp")
cmd.color("yellow", "atp")

# 뷰 조정
cmd.orient("all")

# 고품질 이미지 저장
cmd.ray(1920, 1080)
cmd.png("kinase_visualization.png")

print("Visualization complete!")
```

### 11.2 다중 구조 정렬 스크립트

```python
#!/usr/bin/env python
"""align_structures.py - 여러 구조 정렬 및 비교"""

from pymol import cmd
import glob

def align_all_to_reference(reference_pdb, pattern):
    """모든 구조를 기준 구조에 정렬"""
    
    # 기준 구조 로드
    cmd.load(reference_pdb, "reference")
    
    # 패턴에 맞는 파일 로드 및 정렬
    files = glob.glob(pattern)
    rmsd_results = {}
    
    for pdb_file in files:
        name = pdb_file.split("/")[-1].replace(".pdb", "")
        if name == "reference":
            continue
            
        cmd.load(pdb_file, name)
        
        # CE 정렬 수행
        result = cmd.cealign("reference", name)
        rmsd_results[name] = result.get("RMSD", "N/A")
        
    # 결과 출력
    print("\n=== Alignment Results ===")
    for name, rmsd in rmsd_results.items():
        print(f"{name}: RMSD = {rmsd:.2f} Å")
    
    # 시각화
    cmd.show("cartoon", "all")
    cmd.color("gray", "reference")
    cmd.set("cartoon_transparency", 0.5, "reference")
    
    return rmsd_results

# 사용 예
# align_all_to_reference("reference.pdb", "structures/*.pdb")
```

### 11.3 자동 회전 애니메이션

```python
#!/usr/bin/env python
"""rotation_animation.py - 360도 회전 애니메이션"""

from pymol import cmd
import time

def create_rotation_movie(pdb_id, frames=360, output_prefix="frame"):
    """360도 회전 프레임 생성"""
    
    cmd.fetch(pdb_id, async_=0)
    cmd.show("cartoon")
    cmd.color("spectrum")
    cmd.orient()
    
    for i in range(frames):
        cmd.turn("y", 360.0 / frames)
        cmd.ray(800, 600)
        cmd.png(f"{output_prefix}_{i:04d}.png")
        print(f"Frame {i+1}/{frames}")
    
    print("Animation frames complete!")
    print(f"Combine with: ffmpeg -framerate 30 -i {output_prefix}_%04d.png -c:v libx264 movie.mp4")

# 사용 예
# create_rotation_movie("1crn")
```

### 11.4 배치 처리 스크립트

```python
#!/usr/bin/env python
"""batch_process.py - 여러 PDB 파일 배치 처리"""

from pymol import cmd
import sys
import os

def batch_render(input_dir, output_dir):
    """디렉토리의 모든 PDB 파일을 렌더링"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for pdb_file in os.listdir(input_dir):
        if not pdb_file.endswith(".pdb"):
            continue
            
        name = pdb_file.replace(".pdb", "")
        input_path = os.path.join(input_dir, pdb_file)
        output_path = os.path.join(output_dir, f"{name}.png")
        
        # 초기화
        cmd.reinitialize()
        
        # 로드 및 시각화
        cmd.load(input_path, name)
        cmd.show("cartoon")
        cmd.color("spectrum")
        cmd.orient()
        
        # 렌더링
        cmd.ray(1920, 1080)
        cmd.png(output_path)
        
        print(f"Rendered: {output_path}")

# Headless 모드로 실행
if __name__ == "__main__":
    import pymol
    pymol.finish_launching(['pymol', '-qc'])  # quiet, command-line mode
    
    if len(sys.argv) == 3:
        batch_render(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python batch_process.py <input_dir> <output_dir>")
```

### 11.5 활성 부위 분석 스크립트

```python
#!/usr/bin/env python
"""active_site_analysis.py - 활성 부위 시각화 및 분석"""

from pymol import cmd

def analyze_active_site(pdb_id, ligand_resn, distance=5.0):
    """리간드 주변 활성 부위 분석"""
    
    # 구조 로드
    cmd.fetch(pdb_id, async_=0)
    
    # 리간드 선택
    cmd.select("ligand", f"resn {ligand_resn}")
    
    # 활성 부위 선택 (리간드 주변 잔기)
    cmd.select("active_site", f"byres within {distance} of ligand")
    
    # 시각화 설정
    cmd.hide("all")
    cmd.show("cartoon", "all")
    cmd.color("gray", "all")
    cmd.set("cartoon_transparency", 0.7)
    
    # 활성 부위 강조
    cmd.show("sticks", "active_site")
    cmd.color("cyan", "active_site and elem C")
    
    # 리간드 강조
    cmd.show("sticks", "ligand")
    cmd.color("yellow", "ligand and elem C")
    
    # 수소 결합 표시
    cmd.distance("hbonds", "ligand", "active_site", mode=2)
    cmd.hide("labels", "hbonds")
    cmd.color("red", "hbonds")
    
    # 표면 표시
    cmd.show("surface", "active_site")
    cmd.set("surface_color", "white", "active_site")
    cmd.set("transparency", 0.3, "active_site")
    
    # 뷰 조정
    cmd.orient("ligand")
    cmd.zoom("ligand", 10)
    
    # 잔기 목록 출력
    stored_residues = []
    cmd.iterate("active_site and name CA", "stored_residues.append((resn, resi, chain))", 
                space={'stored_residues': stored_residues})
    
    print(f"\n=== Active Site Residues (within {distance}Å of {ligand_resn}) ===")
    for resn, resi, chain in sorted(set(stored_residues)):
        print(f"  {chain}:{resn}{resi}")
    
    return stored_residues

# 사용 예
# analyze_active_site("1atp", "ATP", 5.0)
```

---

## 12. FoldMason + PyMOL 통합 워크플로우

### 12.1 전체 파이프라인

```bash
# 1. FoldMason으로 다중 구조 정렬
foldmason easy-msa structures/*.pdb alignment tmp/ --report-mode 1

# 2. 정렬된 구조를 PyMOL로 시각화
pymol -qc visualize_alignment.py
```

### 12.2 통합 시각화 스크립트

```python
#!/usr/bin/env python
"""visualize_foldmason_alignment.py - FoldMason 정렬 결과 시각화"""

from pymol import cmd
import os

def visualize_foldmason_results(pdb_dir, alignment_html=None):
    """FoldMason 정렬 결과를 PyMOL에서 시각화"""
    
    # 모든 PDB 파일 로드
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(('.pdb', '.cif'))]
    
    colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    for i, pdb_file in enumerate(pdb_files):
        name = pdb_file.split('.')[0]
        cmd.load(os.path.join(pdb_dir, pdb_file), name)
        
        # 색상 적용
        color = colors[i % len(colors)]
        cmd.color(color, name)
    
    # 첫 번째 구조를 기준으로 정렬
    reference = pdb_files[0].split('.')[0]
    for pdb_file in pdb_files[1:]:
        name = pdb_file.split('.')[0]
        cmd.cealign(reference, name)
    
    # 시각화 설정
    cmd.show("cartoon", "all")
    cmd.orient("all")
    
    # 고품질 이미지 저장
    cmd.ray(1920, 1080)
    cmd.png("foldmason_alignment.png")
    
    print(f"Loaded and aligned {len(pdb_files)} structures")

# 실행
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_foldmason_results(sys.argv[1])
```

### 12.3 LDDT 점수 기반 색상 표시

```python
#!/usr/bin/env python
"""color_by_lddt.py - LDDT 점수로 구조 색상 지정"""

from pymol import cmd

def color_by_lddt(pdb_file, lddt_file):
    """LDDT 점수에 따라 잔기 색상 지정"""
    
    # 구조 로드
    cmd.load(pdb_file, "structure")
    
    # LDDT 점수 로드
    lddt_scores = {}
    with open(lddt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                resi = int(parts[0])
                score = float(parts[1])
                lddt_scores[resi] = score
    
    # 점수에 따라 색상 지정
    cmd.show("cartoon", "structure")
    
    for resi, score in lddt_scores.items():
        if score >= 0.9:
            color = "blue"       # Very high confidence
        elif score >= 0.7:
            color = "cyan"       # High confidence
        elif score >= 0.5:
            color = "yellow"     # Medium confidence
        else:
            color = "orange"     # Low confidence
        
        cmd.color(color, f"structure and resi {resi}")
    
    cmd.orient()
    print("LDDT coloring applied")
```

---

## 13. 참고 자료

### FoldMason

| 리소스 | URL |
|--------|-----|
| GitHub | https://github.com/steineggerlab/foldmason |
| 웹서버 | https://search.foldseek.com/foldmason |
| 공식 사이트 | https://foldmason.foldseek.com |
| 논문 | https://www.science.org/doi/10.1126/science.ads6733 |
| bioRxiv | https://www.biorxiv.org/content/10.1101/2024.08.01.606130 |
| Bioconda | https://bioconda.github.io/recipes/foldmason/README.html |

### PyMOL

| 리소스 | URL |
|--------|-----|
| 공식 사이트 | https://pymol.org |
| PyMOL Wiki | https://pymolwiki.org |
| Python API | https://pymol.org/dokuwiki/doku.php?id=api |
| 스크립팅 튜토리얼 | https://pymol.org/tutorials/scripting/index.html |
| Wikipedia | https://en.wikipedia.org/wiki/PyMOL |

### 관련 도구

| 도구 | URL | 설명 |
|------|-----|------|
| Foldseek | https://github.com/steineggerlab/foldseek | 빠른 구조 검색 |
| MMseqs2 | https://github.com/soedinglab/MMseqs2 | 서열 검색/클러스터링 |
| AlphaFold | https://alphafold.ebi.ac.uk | 구조 예측 |
| ChimeraX | https://www.cgl.ucsf.edu/chimerax/ | 분자 시각화 |
| VMD | https://www.ks.uiuc.edu/Research/vmd/ | 분자 동역학 시각화 |

---

## 부록: 빠른 참조 카드

### FoldMason 명령어

```bash
# 기본 다중 정렬
foldmason easy-msa *.pdb result tmp/

# HTML 리포트 포함
foldmason easy-msa *.pdb result tmp/ --report-mode 1

# 대규모 (사전 클러스터링)
foldmason easy-msa *.pdb result tmp/ --precluster

# MSA 개선
foldmason refinemsa db result.fa refined.fa --refine-iters 3

# 데이터베이스 생성
foldmason createdb *.pdb myDB
```

### PyMOL 명령어

```
fetch 1atp              # PDB 로드
show cartoon            # Cartoon 표시
color spectrum          # 스펙트럼 색상
select sel, chain A     # 선택
orient                  # 뷰 조정
ray 1920, 1080         # Ray-trace
png output.png         # PNG 저장
cealign ref, mobile    # 구조 정렬
distance d, s1, s2     # 거리 측정
```

---

*문서 작성일: 2026-02-05*
*버전: 1.0*
