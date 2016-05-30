#install.packages("rmarkdown", type = "source")

# Render the entire site.
#rmarkdown::render_site()

#render_site(input = ".", output_format = "all", envir = parent.frame(),quiet = FALSE, encoding = getOption("encoding"))

# Render a single file only.
#rmarkdown::render_site("about.Rmd")

# List which files will be removed.
#rmarkdown::clean_site(preview = TRUE)

# Actually remove the files.
rmarkdown::clean_site()

