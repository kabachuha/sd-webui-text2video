function submit_txt2vid(){
    // rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('text2vid', false)

    var id = randomId()
    // Using progressbar without the gallery
    requestProgress(id, null, null, function(){
        showSubmitButtons('text2vid', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}
