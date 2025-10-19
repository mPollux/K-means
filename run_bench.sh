#!/usr/bin/env bash
set -euo pipefail

# ===================== COMPILAÇÃO =====================
echo "==> Compilando kmeans_1d_naive.c (sem OpenMP, usa clock())"
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm

echo "==> Compilando kmeans_1d_omp.c (com OpenMP)"
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

# Verificações
[ -x ./kmeans_1d_naive ] || { echo "ERRO: kmeans_1d_naive não gerado."; exit 1; }
[ -x ./kmeans_1d_omp ]   || { echo "ERRO: kmeans_1d_omp não gerado.";   exit 1; }

# ===================== CONFIGURAÇÃO =====================
# Conjuntos conforme seus nomes:
DATASETS=("p" "m" "g")
DADOS=("dados_p.csv" "dados_m.csv" "dados_g.csv")
CENTR=("centroides_iniciais_p.csv" "centroides_iniciais_m.csv" "centroides_iniciais_g.csv")

MAX_ITER=50
EPS="1e-6"

THREADS=(1 2 4 8 16)

# 1) Schedules base (sem chunk explícito)
SCHEDULES_BASE=("static" "dynamic")
# 2) Afinar chunk para estes schedules
SCHEDULES_TUNE=("static" "dynamic")
CHUNKS=(1 64 256 1024)

REPS=5

STAMP=$(date +"%Y%m%d_%H%M%S")
OUTCSV="resultados_${STAMP}.csv"

# ===================== FUNÇÕES AUXILIARES =====================
parse_line() {
  # Extrai: iters, sse_final, time_ms da linha padrão
  awk '
    /Iterações:/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Iterações:/)   { iters=$(i+1) }
        if ($i ~ /^SSE/)          { sse=$(i+2) }
        if ($i ~ /^Tempo:/)       { t=$(i+1) }
      }
      gsub(/\r/,"",iters); gsub(/\r/,"",sse); gsub(/\r/,"",t);
      print iters, sse, t
    }
  '
}

median() {
  awk '{a[NR]=$1} END{
    if (NR==0) {print "NA"; exit}
    n=asort(a)
    if (n%2==1) print a[(n+1)/2];
    else print (a[n/2]+a[n/2+1])/2.0
  }'
}

run_one_seq() {
  local dados="$1" centr="$2"
  ./kmeans_1d_naive "$dados" "$centr" "$MAX_ITER" "$EPS" | parse_line
}

run_one_omp() {
  local dados="$1" centr="$2"
  ./kmeans_1d_omp "$dados" "$centr" "$MAX_ITER" "$EPS" | parse_line
}

write_header() {
  echo "dataset,modo,threads,schedule,chunk,rep,iters,sse_final,time_ms,median_ms" > "$OUTCSV"
}

append_csv() {
  local dataset="$1" modo="$2" threads="$3" schedule="$4" chunk="$5" rep="$6"
  local iters="$7" sse="$8" time_ms="$9" median_ms="${10}"
  echo "${dataset},${modo},${threads},${schedule},${chunk},${rep},${iters},${sse},${time_ms},${median_ms}" >> "$OUTCSV"
}

# ===================== EXECUÇÃO =====================
write_header

# 1) BASELINE SEQUENCIAL: 5x por dataset → mediana vira baseline
declare -A TEMPO_SERIAL_MS
for idx in "${!DATASETS[@]}"; do
  ds="${DATASETS[$idx]}"; dados="${DADOS[$idx]}"; centr="${CENTR[$idx]}"
  echo "==> Sequencial: dataset=${ds}"
  times=()
  for rep in $(seq 1 "$REPS"); do
    read iters sse tms < <( run_one_seq "$dados" "$centr" )
    echo "   [seq ${ds}] rep=${rep}  iters=${iters}  SSE=${sse}  time_ms=${tms}"
    append_csv "$ds" "seq" 1 "NA" "NA" "$rep" "$iters" "$sse" "$tms" "NA"
    times+=("$tms")
  done
  med=$(printf "%s\n" "${times[@]}" | median)
  TEMPO_SERIAL_MS["$ds"]="$med"
  echo "   [seq ${ds}] mediana(ms)=${med}"
  append_csv "$ds" "seq" 1 "NA" "NA" 0 "NA" "NA" "NA" "$med"
done

# 2) OPENMP — ESCALONAMENTO
# 2.1) Schedules base (sem chunk)
for idx in "${!DATASETS[@]}"; do
  ds="${DATASETS[$idx]}"; dados="${DADOS[$idx]}"; centr="${CENTR[$idx]}"
  for sched in "${SCHEDULES_BASE[@]}"; do
    export OMP_SCHEDULE="$sched"
    echo "==> OMP: dataset=${ds} schedule=${sched} (sem chunk) — escalonamento em threads"
    for T in "${THREADS[@]}"; do
      export OMP_NUM_THREADS="$T"
      times=()
      for rep in $(seq 1 "$REPS"); do
        read iters sse tms < <( run_one_omp "$dados" "$centr" )
        echo "   [omp ${ds}] T=${T} rep=${rep} iters=${iters} SSE=${sse} time_ms=${tms}"
        append_csv "$ds" "omp" "$T" "$sched" "NA" "$rep" "$iters" "$sse" "$tms" "NA"
        times+=("$tms")
      done
      med=$(printf "%s\n" "${times[@]}" | median)
      append_csv "$ds" "omp" "$T" "$sched" "NA" 0 "NA" "NA" "NA" "$med"
      echo "   [omp ${ds}] T=${T} schedule=${sched} mediana(ms)=${med}  (baseline seq ms=${TEMPO_SERIAL_MS[$ds]})"
    done
  done
done

# 2.2) Afinar chunk (para cada schedule)
for idx in "${!DATASETS[@]}"; do
  ds="${DATASETS[$idx]}"; dados="${DADOS[$idx]}"; centr="${CENTR[$idx]}"
  for sched in "${SCHEDULES_TUNE[@]}"; do
    for chunk in "${CHUNKS[@]}"; do
      export OMP_SCHEDULE="${sched},${chunk}"
      echo "==> OMP: dataset=${ds} schedule=${sched}, chunk=${chunk} — escalonamento em threads"
      for T in "${THREADS[@]}"; do
        export OMP_NUM_THREADS="$T"
        times=()
        for rep in $(seq 1 "$REPS"); do
          read iters sse tms < <( run_one_omp "$dados" "$centr" )
          echo "   [omp ${ds}] T=${T} rep=${rep} iters=${iters} SSE=${sse} time_ms=${tms}"
          append_csv "$ds" "omp" "$T" "$sched" "$chunk" "$rep" "$iters" "$sse" "$tms" "NA"
          times+=("$tms")
        done
        med=$(printf "%s\n" "${times[@]}" | median)
        append_csv "$ds" "omp" "$T" "$sched" "$chunk" 0 "NA" "NA" "NA" "$med"
        echo "   [omp ${ds}] T=${T} schedule=${sched},chunk=${chunk} mediana(ms)=${med}"
      done
    done
  done
done

echo
echo "==> Concluído."
echo "CSV consolidado: ${OUTCSV}"
echo "Colunas: dataset,modo,threads,schedule,chunk,rep,iters,sse_final,time_ms,median_ms"