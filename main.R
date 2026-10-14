# ============================================================
# ITR.Forest 仿真实验脚本（适配实际 API）
# ============================================================

library(ITR.Forest)

# ---------- 1. 输出目录与日志 ----------
out_dir <- file.path(getwd(), "ITR_Forest_Results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
cat("输出目录：", out_dir, "\n")

log_file <- file.path(out_dir, "run_log.txt")
log_con  <- file(log_file, open = "wt")

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste(...))
  cat(msg, "\n")
  writeLines(msg, log_con)
}

log_msg("===== 开始运行 ITR.Forest 仿真实验 =====")

# ---------- 2. 生成仿真数据 ----------
set.seed(42)

n <- 800
X <- matrix(runif(n * 4), ncol = 4)
colnames(X) <- c("X1", "X2", "X3", "X4")

beta1 <- 3
A <- (X[, 1] > 0.3) & (X[, 3] > 0.1)
T <- rbinom(n, size = 1, prob = 0.5)
epsilon <- rnorm(n, 0, 1)

Y <- 2 + 2 * (X[, 2] < 0.5) + beta1 * A * T + 1 * (1 - A) * (1 - T) + epsilon

# 关键：ITR.Forest 要求列名为 y, trt, prtx
dat <- data.frame(y = Y, trt = T, prtx = 0.5, X)

log_msg("仿真数据生成完成：n =", n, "，真实 beta1 =", beta1)

write.csv(dat, file.path(out_dir, "simulation_data.csv"), row.names = FALSE)
log_msg("已保存：simulation_data.csv")

# ---------- 3. 训练 ITR-Tree ----------
log_msg("开始训练 ITR-Tree ...")

fit_tree <- tryCatch({
  grow.ITR(
    data      = dat,
    split.var = 4:7,      # X1-X4 在第 4-7 列
    min.ndsz  = 20,
    n0        = 5,
    max.depth = 10
  )
}, error = function(e) {
  log_msg("grow.ITR 训练失败：", conditionMessage(e))
  NULL
})

if (!is.null(fit_tree)) {
  sink(file.path(out_dir, "ITRtree_summary.txt"))
  print(fit_tree)
  sink()
  log_msg("已保存：ITRtree_summary.txt")
}

# ---------- 4. 训练 ITR-Forest ----------
log_msg("开始训练 ITR-Forest（500 棵树）...")

fit_forest <- tryCatch({
  Build.RF.ITR(
    dat       = dat,
    col.y     = "y",
    col.trt   = "trt",
    col.prtx  = "prtx",
    split.var = 4:7,
    ntree     = 500
  )
}, error = function(e) {
  log_msg("Build.RF.ITR 训练失败：", conditionMessage(e))
  NULL
})

if (is.null(fit_forest)) {
  log_msg("森林训练失败，脚本终止。")
  close(log_con)
  stop("Build.RF.ITR 训练失败")
}
log_msg("ITR-Forest 训练完成")

# ---------- 5. 变量重要性 ----------
log_msg("计算变量重要性 ...")

var_imp <- tryCatch({
  Variable.Importance.ITR(fit_forest, n0 = 2)
}, error = function(e) {
  log_msg("Variable.Importance.ITR 失败：", conditionMessage(e))
  NULL
})

if (!is.null(var_imp)) {
  vi_df <- data.frame(
    Variable   = names(var_imp),
    Importance = as.numeric(var_imp)
  )
  vi_df <- vi_df[order(-vi_df$Importance), ]
  write.csv(vi_df, file.path(out_dir, "variable_importance.csv"), row.names = FALSE)
  log_msg("已保存：variable_importance.csv")
  print(vi_df)

  png(file.path(out_dir, "variable_importance.png"),
      width = 900, height = 600, res = 120)
  par(mar = c(5, 6, 4, 2))
  barplot(
    var_imp,
    horiz = TRUE, las = 1,
    col = "steelblue",
    main = "Variable Importance (ITR-Forest)",
    xlab = "Importance"
  )
  dev.off()
  log_msg("已保存：variable_importance.png")
}

# ---------- 6. 价值函数评估（使用 itrtest） ----------
log_msg("计算价值函数 ...")

# 对训练数据使用“全用 T=1”和“全用 T=0”的价值函数
val_t1 <- tryCatch(
  itrtest(dat, z = rep(1, n), n0 = 5),
  error = function(e) NA
)
val_t0 <- tryCatch(
  itrtest(dat, z = rep(0, n), n0 = 5),
  error = function(e) NA
)

cat("Value (all T=1):", val_t1, "\n")
cat("Value (all T=0):", val_t0, "\n")
log_msg("Value (all T=1) =", val_t1, "，Value (all T=0) =", val_t0)

val_df <- data.frame(
  Metric = c("Value_all_T1", "Value_all_T0", "beta1_true", "n", "ntree"),
  Value  = c(val_t1, val_t0, beta1, n, 500)
)
write.csv(val_df, file.path(out_dir, "value_function.csv"), row.names = FALSE)
log_msg("已保存：value_function.csv")

# ---------- 7. 保存 R 对象 ----------
saveRDS(
  list(
    dat        = dat,
    fit_tree   = fit_tree,
    fit_forest = fit_forest,
    var_imp    = var_imp,
    val_t1     = val_t1,
    val_t0     = val_t0
  ),
  file = file.path(out_dir, "ITR_Forest_objects.rds")
)
log_msg("已保存：ITR_Forest_objects.rds")

# ---------- 8. 保存会话信息 ----------
sink(file.path(out_dir, "sessionInfo.txt"))
print(sessionInfo())
sink()
log_msg("已保存：sessionInfo.txt")

log_msg("===== 实验完成 =====")
close(log_con)

cat("\n所有结果已保存至：", out_dir, "\n")
cat("文件清单：\n")
print(list.files(out_dir))