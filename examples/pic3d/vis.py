#!/usr/bin/env python3

import sys
import vtk


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} file.vti")
        sys.exit(1)

    filename = sys.argv[1]

    # ------------------------------------------------------------
    # Read VTI
    # ------------------------------------------------------------
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    image = reader.GetOutput()

    print("Dimensions :", image.GetDimensions())
    print("Origin     :", image.GetOrigin())
    print("Spacing    :", image.GetSpacing())
    print("Extent     :", image.GetExtent())

    # ------------------------------------------------------------
    # Get NodeVol scalar field
    # ------------------------------------------------------------
    scalars = image.GetPointData().GetArray("nd.e-")

    if scalars is None:
        print("Error: NodeVol array was not found")
        print("Available arrays:")

        point_data = image.GetPointData()

        for i in range(point_data.GetNumberOfArrays()):
            print("  ", point_data.GetArrayName(i))

        sys.exit(1)

    image.GetPointData().SetScalars(scalars)

    scalar_min, scalar_max = scalars.GetRange()

    print("NodeVol range:", scalar_min, scalar_max)

    # ------------------------------------------------------------
    # Volume mapper
    # ------------------------------------------------------------
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(image)

    # ------------------------------------------------------------
    # Color transfer function
    # ------------------------------------------------------------
    color = vtk.vtkColorTransferFunction()

    # Blue -> cyan -> green -> yellow -> red
    color.AddRGBPoint(
        0,
        0.0, 0.0, 0.0
    )

    color.AddRGBPoint(
        0.25 * (scalar_max),
        0.0, 1.0, 1.0
    )

    color.AddRGBPoint(
        0.50 * (scalar_max),
        0.0, 1.0, 0.0
    )

    color.AddRGBPoint(
        0.75 * (scalar_max),
        1.0, 1.0, 0.0
    )

    color.AddRGBPoint(
        scalar_max,
        1.0, 0.8, 0.2
    )

    # ------------------------------------------------------------
    # Opacity transfer function
    # ------------------------------------------------------------
    opacity = vtk.vtkPiecewiseFunction()

    # Make low values transparent and high values opaque
    opacity.AddPoint(0, .0)


    opacity.AddPoint(
        scalar_max * 0.5,
        0.8
    )

    opacity.AddPoint(
        scalar_max,
        1
    )
    # ------------------------------------------------------------
    # Volume properties
    # ------------------------------------------------------------
    volume_property = vtk.vtkVolumeProperty()

    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)

    volume_property.SetInterpolationTypeToLinear()



    # ------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    # ------------------------------------------------------------
    # Renderer
    # ------------------------------------------------------------
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.0, 0.0, 0.0)

    # ------------------------------------------------------------
    # Window
    # ------------------------------------------------------------
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1000, 800)
    window.SetWindowName("VTI Viewer - NodeVol")

    # ------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # ------------------------------------------------------------
    # Start
    # ------------------------------------------------------------
    renderer.ResetCamera()

    window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
