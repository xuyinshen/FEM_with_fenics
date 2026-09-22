#!/bin/bash

# 参数列表（按需修改）
MESH_SIZE_FINE_LIST=(0.05)
RC_LIST=(2.0)
XC_LIST=(3.0)
L_LIST=(15.0)
H_LIST=(10.0)
W_LIST=(20.0)
Inner_LIST=(1.1)
Outer_LIST=(1.3)
v0_LIST=(1.0)

for fine in "${MESH_SIZE_FINE_LIST[@]}"; do
  for rc in "${RC_LIST[@]}"; do
    for xc in "${XC_LIST[@]}"; do
      for L in "${L_LIST[@]}"; do
        for H in "${H_LIST[@]}"; do
          for IN in "${Inner_LIST[@]}"; do
            for OUT in "${Outer_LIST[@]}"; do
              for W in "${W_LIST[@]}"; do
                for v0 in "${v0_LIST[@]}"; do
          # 动态生成临时配置文件
          cat > temp_config.yaml <<EOF
mesh:
  fine: $fine
  R_inner: $IN         
  R_outer: $OUT
geometry:
  rc_factor: $rc
  xc_factor: $xc
  L_factor: $L
  H_factor: $H
  W_factor: $W
physics:
  v0: $v0
EOF

          # 基础日志文件名（避免重复拼接）
                 LOG_BASE="run_fine${fine}_rc${rc}_xc${xc}_L${L}_H${H}_v0${v0}"

                 echo "🚀 正在运行: fine=$fine, rc=$rc, xc=$xc, L=$L, H=$H"
          
          # 分别运行不同核数，日志独立保存，防止覆盖
                 python -u NS_3D_MPI_SLIP.py --config temp_config.yaml 2>&1 | tee "${LOG_BASE}_n1.log"
          #      mpiexec -n 4 python -u NS_3D_MPI_RUN.py --config temp_config.yaml 2>&1 | tee "${LOG_BASE}_n4.log"
          #      mpiexec -n 8 python -u NS_3D_MPI_RUN.py --config temp_config.yaml 2>&1 | tee "${LOG_BASE}_n8.log"
          #      mpiexec -n 12 python -u NS_3D_MPI_SLIP.py --config temp_config.yaml 2>&1 | tee "${LOG_BASE}_n12.log"

          # 清理临时文件
                rm -f temp_config.yaml
                sleep 2  # 释放文件锁与磁盘缓存
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "🎉 批量计算全部结束！"
