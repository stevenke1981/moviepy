"""Packaged compositor readback; no scene-provided scripts or shaders."""

LINEAR_CAPTURE = r'''
extends CompositorEffect

var frame_limit: int = 0
var frame_index: int = 0

func _init():
    effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
    access_resolved_color = true

func _render_callback(_kind: int, data: RenderData):
    if frame_index >= frame_limit:
        return
    var buffers: RenderSceneBuffersRD = data.get_render_scene_buffers()
    var device = RenderingServer.get_rendering_device()
    var texture = buffers.get_color_layer(0)
    var format = device.texture_get_format(texture)
    if format.format != RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT:
        push_error("Linear capture requires native RGBA16F")
        return
    # The internal color texture cannot be downloaded directly. An exact
    # image load/store copies its FP16 values to our own readable texture.
    var copy_format = RDTextureFormat.new()
    copy_format.width = format.width
    copy_format.height = format.height
    copy_format.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
    copy_format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
    var copied = device.texture_create(copy_format, RDTextureView.new())
    var source = RDShaderSource.new()
    source.source_compute = """
#version 450
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
layout(rgba16f, set=0, binding=0) uniform readonly image2D original_image;
layout(rgba16f, set=0, binding=1) uniform writeonly image2D copied_image;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, imageSize(copied_image)))) return;
    imageStore(copied_image, pixel, imageLoad(original_image, pixel));
}
"""
    var shader = device.shader_create_from_spirv(device.shader_compile_spirv_from_source(source))
    var pipeline = device.compute_pipeline_create(shader)
    var input = RDUniform.new()
    input.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
    input.binding = 0
    input.add_id(texture)
    var target = RDUniform.new()
    target.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
    target.binding = 1
    target.add_id(copied)
    var uniforms = device.uniform_set_create([input, target], shader, 0)
    var commands = device.compute_list_begin()
    device.compute_list_bind_compute_pipeline(commands, pipeline)
    device.compute_list_bind_uniform_set(commands, uniforms, 0)
    device.compute_list_dispatch(commands, int(ceil(format.width / 8.0)), int(ceil(format.height / 8.0)), 1)
    device.compute_list_end()
    var raw = device.texture_get_data(copied, 0)
    device.free_rid(uniforms)
    device.free_rid(pipeline)
    device.free_rid(shader)
    device.free_rid(copied)
    if raw.size() != format.width * format.height * 8:
        push_error("Linear capture returned an incomplete frame")
        return
    var image = Image.create_from_data(format.width, format.height, false, Image.FORMAT_RGBAH, raw)
    var path = "res://linear_raw/frame%08d" % frame_index
    if image.save_exr(path + ".exr", false) != OK:
        push_error("Linear EXR export failed")
        return
    var file = FileAccess.open(path + ".json", FileAccess.WRITE)
    file.store_string(JSON.stringify({"frame": frame_index,
        "size": [format.width, format.height], "format": "RGBA16F",
        "stage": "post_transparent_before_postprocessing"}))
    file.close()
    frame_index += 1
'''
