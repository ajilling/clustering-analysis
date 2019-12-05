suppressPackageStartupMessages(library("caret"))

directory = "~/Desktop/Thesis/Data/Test"

fileNames = list.files(directory)

for (name in fileNames){
  print(paste("Filename: ", name))
  mf = read.csv(file = paste0(directory, "/", name), header = FALSE, sep = ",")
  matrix = as.matrix(mf)
  print(findLinearCombos(matrix))

}
