#!/bin/bash
# 硬盘空间监控脚本
# 用法: bash monitor_disk_space.sh

OUTPUT_DIR="/home/suhang/projects/nanoLLaVA/outputs/nano_vtla_baseline"
THRESHOLD=10  # 低于 10GB 时报警

while true; do
    # 检查可用空间
    AVAIL=$(df / | tail -1 | awk '{print $4}')
    AVAIL_GB=$((AVAIL / 1024 / 1024))
    
    # 检查 checkpoint 数量
    NUM_CKPTS=$(ls $OUTPUT_DIR/checkpoint_*.pt 2>/dev/null | wc -l)
    
    # 显示状态
    clear
    echo "========================================="
    echo "Nano-VTLA 训练监控"
    echo "========================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "硬盘状态:"
    df -h / | grep -E "Filesystem|/$"
    echo ""
    echo "Checkpoints: $NUM_CKPTS 个"
    ls -lth $OUTPUT_DIR/checkpoint_*.pt 2>/dev/null | head -5
    echo ""
    
    # 报警
    if [ $AVAIL_GB -lt $THRESHOLD ]; then
        echo "⚠️  警告: 可用空间不足 ${AVAIL_GB}GB！"
        echo "建议: 减少 --keep_checkpoints 或清理其他文件"
    else
        echo "✅ 空间充足: ${AVAIL_GB}GB 可用"
    fi
    
    echo ""
    echo "按 Ctrl+C 停止监控"
    echo "========================================="
    
    # 每30秒更新
    sleep 30
done
