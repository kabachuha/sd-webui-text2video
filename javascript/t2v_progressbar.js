function submit_txt2vid(){
    // rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('txt2vid', false)

    var id = randomId()
    // Using progressbar without the gallery
    requestProgress(id, gradioApp().getElementById('txt2vid_results_panel'), null, function(){
        showSubmitButtons('txt2vid', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}
