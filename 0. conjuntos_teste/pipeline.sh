#!/usr/bin/env bash
set -euo pipefail

echo "=== Gerando conjunto de dados PEQUENO (N=10^4, K=4) ==="
awk 'BEGIN{srand(42);
  for(i=0;i<10000;i++){
    m=(i%4)*10;               # 0,10,20,30
    x=m + (rand()-0.5)*2;     # ruído +-1
    print x
  }
}' > dados_p.csv
printf "0\n10\n20\n30\n" > centroides_iniciais_p.csv

echo "=== Gerando conjunto de dados MÉDIO (N=10^5, K=8) ==="
awk 'BEGIN{srand(4242);
  for(i=0;i<100000;i++){
    m=(i%8)*10;               # 0..70
    x=m + (rand()-0.5)*2;
    print x
  }
}' > dados_m.csv
printf "0\n10\n20\n30\n40\n50\n60\n70\n" > centroides_iniciais_m.csv

echo "=== Gerando conjunto de dados GRANDE (N=10^6, K=16) ==="
awk 'BEGIN{srand(424242);
  for(i=0;i<1000000;i++){
    m=(i%16)*10;              # 0..150
    x=m + (rand()-0.5)*2;
    print x
  }
}' > dados_g.csv
for v in 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do echo $v; done > centroides_iniciais_g.csv

echo "=== Conjuntos de entrada gerados com sucesso ==="
echo "Arquivos: dados_p.csv, dados_m.csv, dados_g.csv e seus respectivos centroides iniciais"
