inherit cmake

DESCRIPTION = "Example of helmet detection"
LICENSE = "BSD"
SECTION = "helmet"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/${LICENSE};md5=3775480a712fc46a69647678acb234cb"

# Dependencies.
DEPENDS := "opencv"
DEPENDS += "zlib"
DEPENDS += "libnl zlib"
DEPENDS += "glib-2.0"
DEPENDS += "libutils"
DEPENDS += "libcutils"

DEPENDS += "gstreamer1.0"
DEPENDS += "gstreamer1.0-plugins-base"
DEPENDS += "gstreamer1.0-plugins-qti-oss-mlmeta"
DEPENDS += "gstreamer1.0-plugins-qti-oss-tools"
DEPENDS += "gstreamer1.0-rtsp-server"

DEPENDS += "fastcv-noship"
DEPENDS += "tensorflow-lite"

FILESPATH =+ "${WORKSPACE}/ai-demo/FaceMaskDetect/bin/:"

SRC_URI = "file://FaceMaskDetect/"
PN = "facemask_detect"
PV = "1"
INSANE_SKIP_${PN}-dev += "ldflags dev-elf dev-deps"
PACKAGES = "${PN}-dbg ${PN} ${PN}-dev"
S = "${WORKSPACE}/ai-demo/FaceMaskDetect/"

# Install directries.
INSTALL_INCDIR := "${includedir}"
INSTALL_BINDIR := "${bindir}"
INSTALL_LIBDIR := "${libdir}"

EXTRA_OECMAKE += ""

FILES_${PN} += "${INSTALL_BINDIR}"
FILES_${PN} += "${INSTALL_LIBDIR}"

SOLIBS = ".so*"
FILES_SOLIBSDEV = ""
