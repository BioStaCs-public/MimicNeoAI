suppressPackageStartupMessages(library(Peptides))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript compute_peptide_features.R input.csv output.csv peptide_col")
}

input_csv <- args[[1]]
output_csv <- args[[2]]
peptide_col <- args[[3]]

data <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
if (!(peptide_col %in% colnames(data))) {
  stop(sprintf("Peptide column not found: %s", peptide_col))
}

calculate_polarity <- function(seq) {
  polarities <- list(
    hydrophilic = c("A", "D", "E", "N", "Q", "R", "S", "T", "Y"),
    hydrophobic = c("I", "L", "M", "F", "P", "V", "W")
  )
  sum(sapply(strsplit(seq, "")[[1]], function(aa) {
    if (aa %in% polarities$hydrophilic) {
      1
    } else if (aa %in% polarities$hydrophobic) {
      -1
    } else {
      0
    }
  }))
}

aa_volumes <- c(
  A = 88.6, C = 118.8, D = 111.1, E = 138.3, F = 189.9, G = 60.1,
  H = 153.2, I = 166.6, K = 168.6, L = 166.6, M = 162.9, N = 114.1,
  P = 115.0, Q = 146.2, R = 174.0, S = 105.9, T = 119.0, V = 140.0,
  W = 227.8, Y = 193.6
)

calculate_volume <- function(seq) {
  sum(sapply(strsplit(seq, "")[[1]], function(aa) {
    if (aa %in% names(aa_volumes)) {
      aa_volumes[[aa]]
    } else {
      0
    }
  }))
}

data[[peptide_col]] <- toupper(trimws(data[[peptide_col]]))
data$aa_composition <- sapply(data[[peptide_col]], function(x) {
  paste(unlist(aaComp(x)), collapse = ", ")
})
data$polarity <- sapply(data[[peptide_col]], calculate_polarity)
data$volume <- sapply(data[[peptide_col]], calculate_volume)
data$net_charge <- sapply(data[[peptide_col]], charge)
data$hydrophobicity <- sapply(data[[peptide_col]], hydrophobicity)
data$boman_index <- sapply(data[[peptide_col]], boman)
data$aliphatic_index <- sapply(data[[peptide_col]], aIndex)
data$isoelectric_point <- sapply(data[[peptide_col]], pI)

write.csv(data, output_csv, row.names = FALSE, quote = TRUE)
