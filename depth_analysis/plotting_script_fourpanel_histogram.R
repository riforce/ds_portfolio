#*
#*This script plots the residual results of the four models.
#*
#*We first read in the four residual files and the test set and
#*add them to a single dataframe. Then, using ggplot2, we construct
#*four subplots and add them to a single figure that is then annotated
#*for clarity.
#*





plotting_script_fourpanel_histogram <- function(input.gam.resids = "C:/Users/riley/ds_portfolio/spline_model_test_resids.csv",
                              input.ols.resids = "C:/Users/riley/ds_portfolio/ml_min_mc_resids.csv",
                              input.fourparamols.resids = "C:/Users/riley/ds_portfolio/fourfeatureOLS_resids.csv",
                              input.baseline.resids = "C:/Users/riley/ds_portfolio/baseline_resids.csv",
                              input.test = "C:/Users/riley/Seismo_Direc/cleaned_testing_df.csv")
{
  
  library(readr)
  library(ggplot2)
  library(ggpubr)
  
  #*
  #*Reading in Data
  #*
  test <- read.csv(input.test)
  depths <- test$Dep
  
  gam.resids <- read.csv(input.gam.resids)
  colnames(gam.resids)[1] ="index"
  colnames(gam.resids)[2] ="resid"
  
  ols.resids <- read.csv(input.ols.resids)
  colnames(ols.resids)[1] ="index"
  colnames(ols.resids)[2] ="resid"
  
  ols4.resids <- read.csv(input.fourparamols.resids)
  colnames(ols4.resids)[1] ="index"
  colnames(ols4.resids)[2] ="resid"
  
  baseline.resids <- read.csv(input.baseline.resids)
  colnames(baseline.resids)[1] ="index"
  colnames(baseline.resids)[2] ="resid"
  
  df <- data.frame(depths, gam.resids$resid, ols.resids$resid, ols4.resids$resid, baseline.resids$resid)
  colnames(df)[2] = "gam_resids"
  colnames(df)[3] = "ols_resids"
  colnames(df)[4] = "ols4_resids"
  colnames(df)[5] = "baseline_resids"
  #print(head(df))
  
  #*
  #* Set up plots
  #*
  
  #set up the percentile lines
  x_dat <- min(depths):max(depths)
  percentiles_baseline <- quantile(df$baseline_resids, probs = c(.1, 0.5, .9))
  percentiles_ols <- quantile(df$ols_resids, probs = c(.1, 0.5, .9))
  percentiles_4ols <- quantile(df$ols4_resids, probs = c(0.1, 0.5, 0.9))
  percentiles_gam <- quantile(df$gam_resids, probs = c(0.1, 0.5, 0.9))
  
  
  p1 <- ggplot(data = df, mapping = aes(x = depths, y = baseline_resids)) +
    geom_point(alpha = 0.08, color = "darkblue") +
    rremove("ylab") + rremove("xlab") +
    ggtitle("Baseline") +
    theme(axis.text=element_text(size=16),
          title=element_text(size=16)) +
    ylim(-14, 24)+
    geom_hline(yintercept=percentiles_baseline[1], linetype="dashed", color = "orangered") +
    geom_hline(yintercept=percentiles_baseline[2], linetype="dashed", color = "violetred2") +
    geom_hline(yintercept=percentiles_baseline[3], linetype="dashed", color = "darkorchid") 
  
  
  p2 <- ggplot(data = df, mapping = aes(x = depths, y = ols_resids)) +
    geom_point(alpha = 0.08, color = "darkblue") +
    rremove("ylab") + rremove("xlab") +
    ggtitle("ML - Mc2") +
    theme(axis.text=element_text(size=16),
          title=element_text(size=16)) +
    ylim(-14,24)+
    geom_hline(yintercept=percentiles_ols[1], linetype="dashed", color = "orangered") +
    geom_hline(yintercept=percentiles_ols[2], linetype="dashed", color = "violetred2") +
    geom_hline(yintercept=percentiles_ols[3], linetype="dashed", color = "darkorchid") 
  
  
  p3 <- ggplot(data = df, mapping = aes(x = depths, y = ols4_resids)) +
    geom_point(alpha = 0.08, color = "darkblue") +
    rremove("ylab") + rremove("xlab") +
    ggtitle("Four-Feature Least Squares") +
    theme(axis.text=element_text(size=16),
          title=element_text(size=16))+
    ylim(-14,24)+
    geom_hline(yintercept=percentiles_4ols[1], linetype="dashed", color = "orangered") +
    geom_hline(yintercept=percentiles_4ols[2], linetype="dashed", color = "violetred2") +
    geom_hline(yintercept=percentiles_4ols[3], linetype="dashed", color = "darkorchid") 
  
  
  p4 <- ggplot(data = df, mapping = aes(x = depths, y = gam_resids)) +
    geom_point(alpha = 0.08, color = "darkblue") +
    rremove("ylab") + rremove("xlab") +
    ggtitle("Natural Spline GAM") +
    theme(axis.text=element_text(size=16),
          title=element_text(size=16)) +
    ylim(-14,24)+
    geom_hline(yintercept=percentiles_gam[1], linetype="dashed", color = "orangered") +
    geom_hline(yintercept=percentiles_gam[2], linetype="dashed", color = "violetred2") +
    geom_hline(yintercept=percentiles_gam[3], linetype="dashed", color = "darkorchid") 
  
  
  fig <- ggarrange(p1, p2, p3, p4, 
                   labels = c("A", "B", "C", "D"),
                   ncol = 2, nrow = 2, 
                   align = "hv",
                   font.label=list(color="blue",size=22)) + theme(aspect.ratio = 0.7)
  
  
  fig <- annotate_figure(fig, top = text_grob("Model Comparison - Residuals", 
                                              face = "bold", size = 35, vjust = 1)) 
  
  fig <- annotate_figure(fig, left = text_grob("Depth Residual (km)", rot = 90, size = 24, vjust = 0),
                         bottom = text_grob("Event Depth (km)", size = 24, vjust = 2))
  
  
  
  fig <- annotate_figure(fig, 
                         bottom = text_grob("---- : 10th percentile", size = 15, color = "orangered", vjust = -10, hjust = 1.5 ))
  fig <- annotate_figure(fig,
                         bottom = text_grob("---- : 50th percentile", size = 15, color = "violetred2", vjust = -15.5, hjust = 0))
  fig4 <- annotate_figure(fig,
                          bottom = text_grob("---- : 90th percentile", size = 15, color = "darkorchid", vjust = -21.5, hjust = -1.5))
  
  
  #Make a histogram of the baseline and gam residuals
  df_base <- data.frame(residual = df$baseline_resids)
  df_gam <- data.frame(residual = df$gam_resids)
  
  df_base$Model <- 'Baseline'
  df_gam$Model <- 'Natural Spline GAM'
  
  combo <- rbind(df_base, df_gam)
  
  hist <- ggplot(combo, aes(x=residual, color=Model)) +
    geom_histogram(fill="white", alpha=0.02, binwidth = 0.2, position = "identity")+
    ggtitle("Model Comparison - Residual Histogram") +
    ylab("Count") +
    xlab("Depth Residual (km)") +
    theme(plot.title = element_text(hjust = 0.5, size = 35, face = "bold"),
          text = element_text(size = 30),
          legend.title = element_text(hjust = 0.5))
  
  hist
  
  
}