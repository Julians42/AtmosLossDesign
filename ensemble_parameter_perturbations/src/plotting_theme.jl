# Shared CairoMakie theme. Only plot scripts should include this -
# postprocessing/analysis scripts have no reason to load a plotting theme.
#
# Expects the including script to have already done: using CairoMakie

function set_default_plot_theme!()
    set_theme!(theme_latexfonts(),
        fontsize = 14,
        fonts = (
            regular = "Latin Modern Roman",
            bold = "Latin Modern Sans Demi Bold",
            italic = "Latin Modern Roman Italic",
            bold_italic = "Latin Modern Roman Bold Italic",
        ),
        Axis = (
            titlefont = "Latin Modern Roman",
            titlesize = 16,
        ),
        Figure = (
            titlefont = :bold,
            titlesize = 18,
        ),
        linewidth = 1.5,
        markersize = 8,
    )
end
