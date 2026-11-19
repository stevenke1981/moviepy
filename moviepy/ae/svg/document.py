"""Validate a deliberately local, static SVG subset before native parsing."""

import copy
import re
import xml.etree.ElementTree as ET

from moviepy.ae.properties.values import finite_real


MAX_SVG_BYTES = 2 * 1024 * 1024
SVG_NS = "http://www.w3.org/2000/svg"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ELEMENTS = frozenset(
    "svg g defs rect circle ellipse line polyline polygon path linearGradient "
    "radialGradient stop clipPath mask text tspan title desc".split()
)
PRESENTATION = frozenset(
    "fill fill-opacity fill-rule stroke stroke-opacity stroke-width "
    "stroke-linecap stroke-linejoin stroke-miterlimit stroke-dasharray "
    "stroke-dashoffset opacity color stop-color stop-opacity clip-path "
    "clip-rule mask display visibility font-family font-size font-style "
    "font-weight font-stretch text-anchor letter-spacing word-spacing "
    "text-decoration dominant-baseline alignment-baseline baseline-shift "
    "shape-rendering text-rendering paint-order".split()
)
ATTRIBUTES = PRESENTATION | frozenset(
    "id x y x1 y1 x2 y2 dx dy width height rx ry r cx cy fx fy fr d points "
    "viewBox preserveAspectRatio transform gradientTransform gradientUnits "
    "spreadMethod offset clipPathUnits maskUnits maskContentUnits rotate "
    "textLength lengthAdjust version".split()
)
PAINT = frozenset({"fill", "stroke", "color", "stop-color"})
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z")
LOCAL_URL = re.compile(
    r"url\(\s*(['\"]?)#([A-Za-z_][A-Za-z0-9_.:-]{0,127})\1\s*\)\Z", re.I
)


class _TreeBuilder(ET.TreeBuilder):
    """Reject entities, processing instructions and excessive trees early."""

    def __init__(self):
        super().__init__()
        self.depth = self.nodes = 0

    def start(self, tag, attrs):
        self.depth += 1
        self.nodes += 1
        if self.depth > 64 or self.nodes > 10000 or len(attrs) > 100:
            raise ValueError("SVG tree exceeds the depth/node/attribute budget")
        return super().start(tag, attrs)

    def end(self, tag):
        self.depth -= 1
        return super().end(tag)

    def doctype(self, name, pubid, system):
        raise ValueError("SVG DTDs and entities are not supported")

    def pi(self, target, text):
        raise ValueError("SVG processing instructions are not supported")


def _name(tag):
    if tag.startswith("{" + SVG_NS + "}"):
        return tag[len(SVG_NS) + 2 :]
    return tag


def _attribute(name, value):
    if name not in ATTRIBUTES:
        raise ValueError(f"unsupported SVG attribute: {name}")
    if any(token in value for token in ("\\", "@", "/*", "*/")):
        raise ValueError("SVG CSS escapes, comments and imports are not supported")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ValueError("invalid control character in SVG attribute")
    if name == "id" and not IDENTIFIER.fullmatch(value):
        raise ValueError("SVG ids must be simple, nonempty identifiers")
    if re.search(r"url\s*\(", value, re.I):
        if name not in {
            "fill",
            "stroke",
            "clip-path",
            "mask",
        } or not LOCAL_URL.fullmatch(value):
            raise ValueError("SVG resources must be local fragment references")
    elif name in {"clip-path", "mask"} and value != "none":
        raise ValueError("SVG resources must be local fragment references")


def parse_svg(svg):
    """Return a detached validated tree; never follow SVG resource references."""
    if not isinstance(svg, str):
        raise TypeError("svg must be XML text; use SVGLayer.from_file for a local file")
    if len(svg.encode("utf-8")) > MAX_SVG_BYTES:
        raise ValueError("SVG exceeds the 2 MiB input budget")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", svg, re.I):
        raise ValueError("SVG DTDs and entities are not supported")
    try:
        root = ET.fromstring(svg, parser=ET.XMLParser(target=_TreeBuilder()))
    except ET.ParseError as error:
        raise ValueError(f"invalid SVG XML: {error}") from error
    if _name(root.tag) != "svg":
        raise ValueError("SVG document must have an svg root")
    ids = set()
    references = []
    for node in root.iter():
        node.tag = _name(node.tag)
        if node.tag not in ELEMENTS:
            raise ValueError(f"unsupported SVG element: {node.tag}")
        # A small inline presentation subset supports ordinary exported shapes.
        # Stylesheets, URL loading, CSS escapes and active content are rejected.
        style = node.attrib.pop("style", None)
        if style is not None:
            for declaration in style.split(";"):
                if not declaration.strip():
                    continue
                key, separator, value = declaration.partition(":")
                key, value = key.strip(), value.strip()
                if not separator or key not in PRESENTATION:
                    raise ValueError("unsupported SVG inline style declaration")
                _attribute(key, value)
                node.set(key, value)
        for key, value in node.attrib.items():
            if key == "{" + XML_NS + "}space" and value in {"default", "preserve"}:
                continue
            _attribute(key, value)
            reference = LOCAL_URL.fullmatch(value)
            if reference is not None:
                references.append(reference.group(2))
        ident = node.get("id")
        if ident is not None:
            if ident in ids:
                raise ValueError(f"duplicate SVG id: {ident}")
            ids.add(ident)
    if any(ref not in ids for ref in references):
        raise ValueError("SVG contains an unresolved local fragment reference")
    return root


def binding_target(root, element_id, attribute):
    """Validate an editable leaf text or presentation/geometry attribute."""
    if not isinstance(element_id, str) or not isinstance(attribute, str):
        raise TypeError("SVG binding id and attribute must be strings")
    node = next((item for item in root.iter() if item.get("id") == element_id), None)
    if node is None:
        raise ValueError(f"SVG element id not found: {element_id}")
    if attribute == "text":
        if node.tag not in {"text", "tspan"} or len(node):
            raise ValueError("text bindings require a leaf text or tspan element")
    elif attribute not in ATTRIBUTES or attribute == "id":
        raise ValueError(f"unsupported editable SVG attribute: {attribute}")
    elif node is root and attribute in {"width", "height"}:
        raise ValueError("set layer.size to change the SVG raster viewport")
    return node


def _binding_value(attribute, prop, value):
    if attribute == "text":
        if not isinstance(value, str):
            raise TypeError("SVG text bindings require string enum Properties")
        return value
    if prop.value_type == "color" and attribute in PAINT:
        if any(not 0 <= component <= 1 for component in value):
            raise ValueError("SVG colors must be SDR components in 0..1")
        rgb = ",".join(str(round(component * 255)) for component in value[:3])
        return f"rgb({rgb})" if len(value) == 3 else f"rgba({rgb},{value[3]:.17g})"
    if isinstance(value, str):
        return value
    if prop.value_type == "float":
        return format(finite_real(value, "SVG property"), ".17g")
    raise TypeError(
        "SVG attributes accept float, string enum, or paint color Properties"
    )


def evaluate_svg(template, bindings, size, t, context, expression_bindings):
    """Apply evaluated Properties with XML escaping, then revalidate the result."""
    root = copy.deepcopy(template)
    for (element_id, attribute), prop in bindings.items():
        node = binding_target(root, element_id, attribute)
        value = _binding_value(
            attribute, prop, prop.value_at(t, context=context, **expression_bindings)
        )
        if attribute == "text":
            node.text = value
        else:
            _attribute(attribute, value)
            node.set(attribute, value)
    root.set("width", str(size[0]))
    root.set("height", str(size[1]))
    root.set("xmlns", SVG_NS)
    serialized = ET.tostring(root, encoding="unicode")
    # This also enforces the serialized size and fragment checks after edits.
    checked = parse_svg(serialized)
    return serialized, checked
