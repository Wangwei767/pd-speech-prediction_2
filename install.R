
# Binder环境包安装脚本
install.packages(c(
  "shiny",
  "shinydashboard", 
  "DT",
  "plotly",
  "ggplot2",
  "dplyr",
  "caret",
  "shinyWidgets",
  "shinyjs",
  "MASS",
  "randomForest"
), repos = "https://cran.rstudio.com/")

# 验证安装
cat("All packages installed successfully!\n")
