import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import SimpleITK as sitk
import sitkUtils


#
# SegmentLungs
#


class SegmentLungs(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SegmentLungs")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "OpenCVAI")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SegmentLungs">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SegmentLungs1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SegmentLungs",
        sampleName="SegmentLungs1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SegmentLungs1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SegmentLungs1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SegmentLungs1",
    )

    # SegmentLungs2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SegmentLungs",
        sampleName="SegmentLungs2",
        thumbnailFileName=os.path.join(iconsPath, "SegmentLungs2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SegmentLungs2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SegmentLungs2",
    )


#
# SegmentLungsParameterNode
#


@parameterNodeWrapper
class SegmentLungsParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# SegmentLungsWidget
#


class SegmentLungsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentLungs.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.outputLabel = self.ui.outputLabel

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SegmentLungsLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[SegmentLungsParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            # self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(), self.ui.referenceSelector.currentNode())
            dice_left, dice_right = self.logic.process(
                self.ui.inputSelector.currentNode(),
                self.ui.outputSelector.currentNode(),
                self.ui.referenceSelector.currentNode()
            )
            
            left_s = f"Dice Coefficient (Left Lung): {dice_left:.4f}\n"
            right_s = f"Dice Coefficient (Right Lung): {dice_right:.4f}\n"
            full_s = left_s + right_s
            self.outputLabel.setText(full_s)



#
# SegmentLungsLogic
#


class SegmentLungsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        import numpy as np
        import os
        try:
            import nibabel as nib
        except ImportError:
            slicer.util.pip_install("nibabel")
            import nibabel as nib
        try:
            from scipy import ndimage as ndi
        except ImportError:
            slicer.util.pip_install("scipy")
            from scipy import ndimage as ndi
        try:
            from skimage import morphology, segmentation, measure, filters
            from skimage.feature import peak_local_max
        except ImportError:
            slicer.util.pip_install("scikit-image")
            from skimage import morphology, segmentation, measure, filters
            from skimage.feature import peak_local_max
        try:
            from surface_distance import metrics
        except ImportError:
            slicer.util.pip_install("surface-distance")
            from surface_distance import metrics
        
        self.os = os
        self.np = np
        self.nib = nib
        self.ndi = ndi
        self.morphology = morphology
        self.segmentation = segmentation
        self.measure = measure
        self.filters = filters
        self.peak_local_max = peak_local_max
        self.metrics = metrics

    def getParameterNode(self):
        return SegmentLungsParameterNode(super().getParameterNode())
    
    def segment_body_slice(self, slice_img, threshold=-191): # -191
    # Apply thresholding
        body = slice_img > threshold

        # Remove small objects
        body = self.morphology.remove_small_objects(body, min_size=3000)

        # Fill holes
        body = self.ndi.binary_fill_holes(body)

        # Perform morphological closing to smooth edges
        struct_elem = self.morphology.disk(5)
        body = self.morphology.closing(body, struct_elem)

        return body.astype(self.np.uint8)
    
    def segment_airway(self, ct_data, lung_mask, lower_boundary, air_threshold=-900):
    # Find threshold based on lung air values
        ct_data_filtered = self.np.where(lung_mask.astype(bool), ct_data, 0)
        print(f"Determined airway threshold: {air_threshold:.2f} HU")

        airway_mask = ct_data_filtered < air_threshold
        
        # Focus on central region where airways are typically located
        center_y = airway_mask.shape[1] // 2
        width = airway_mask.shape[1] // 5 
        airway_mask[:, :center_y-width, :] = 0
        airway_mask[:, center_y+width:, :] = 0

        # Exclude the lower part of the lungs
        airway_mask[:, :, :lower_boundary] = 0

        # Erode the airway mask
        airway_mask = self.morphology.binary_erosion(airway_mask, self.morphology.ball(3))
        
        # Remove small disconnected components
        airway_mask = self.morphology.remove_small_objects(airway_mask, min_size=150)
        
        # Apply morphological operations to clean up the segmentation
        airway_mask = self.morphology.binary_dilation(airway_mask, self.morphology.ball(5))
        
        # Get the largest connected component (main airway)
        labeled_airways, _ = self.ndi.label(airway_mask)
        sizes = self.np.bincount(labeled_airways.ravel())[1:]
        largest_component = sizes.argmax() + 1
        airway_mask = labeled_airways == largest_component
        
        return airway_mask

    def compute_dice_coefficient(self, segmentation, reference):
        return self.metrics.compute_dice_coefficient(segmentation, reference)


    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                referenceVolume: vtkMRMLScalarVolumeNode,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        inputVolumeArray = slicer.util.arrayFromVolume(inputVolume)
        # Initialize the body mask array
        body_mask = self.np.zeros(inputVolumeArray.shape, dtype=self.np.uint8)

        # Iterate over each slice
        for i in range(inputVolumeArray.shape[2]):
            slice_img = inputVolumeArray[:, :, i]
            body_mask[:, :, i] = self.segment_body_slice(slice_img)

        air_threshold = -320
        air_mask = inputVolumeArray < air_threshold

        lung_mask = air_mask & body_mask

        # Calculate cumulative sum along z-axis
        z_sums = self.np.sum(lung_mask, axis=(0,1))
        cumsum = self.np.cumsum(z_sums)

        # Normalize cumulative sum to get percentiles
        cumsum_normalized = cumsum / cumsum[-1]

        z_percentile = self.np.where(cumsum_normalized >= 0.1)[0][0]
        print(f"10th percentile z index: {z_percentile}")

        lower_boundary = z_percentile

        airway_mask = self.segment_airway(inputVolumeArray, lung_mask, lower_boundary)
        lung_mask, airway_mask = lung_mask & ~airway_mask, lung_mask & airway_mask

        # Apply different processing for upper and lower regions
        lung_mask_upper = lung_mask[:, :, lower_boundary:]
        lung_mask_lower = lung_mask[:, :, :lower_boundary]

        # Process upper region (minimal erosion)
        lung_mask[:, :, lower_boundary:] = self.morphology.remove_small_objects(lung_mask_upper, min_size=10000)

        # Process lower region (more aggressive erosion)
        lung_mask[:, :, :lower_boundary] = self.morphology.remove_small_objects(lung_mask_lower, min_size=10000)
        lung_mask[:, :, :lower_boundary] = self.morphology.binary_erosion(lung_mask[:, :, :lower_boundary], self.morphology.ball(3))

        half = self.np.copy(lung_mask[:, lung_mask.shape[1]//2 - 3 : lung_mask.shape[1]//2 + 3, :])
        lung_mask[:, lung_mask.shape[1]//2 - 3 : lung_mask.shape[1]//2 + 3, :] = 0

        # Label connected components
        labeled_lung_mask, num_labels = self.ndi.label(lung_mask)

        # Find the largest two connected components (assuming they are the lungs)
        sizes = self.ndi.sum(lung_mask, labeled_lung_mask, range(1, num_labels + 1))
        largest_labels = self.np.argsort(sizes)[-2:] + 1  # Get label indices of the two largest components

        masks = []
        for label in largest_labels:
            mask = labeled_lung_mask == label
            masks.append(mask)

        # Create a mask for the largest components
        lung_mask_cleaned = self.np.zeros_like(lung_mask, dtype=self.np.uint8)
        for mask in masks:
            lung_mask_cleaned = lung_mask_cleaned | mask

        # restore strip (but only for the upper part of the lungs to avoid restoring bowels)
        lung_mask_cleaned[:, lung_mask_cleaned.shape[1]//2 - 3 : lung_mask_cleaned.shape[1]//2 + 3, lower_boundary:] = half[:, :, lower_boundary:]

        lung_mask_cleaned = self.morphology.remove_small_objects(lung_mask_cleaned, min_size=10000)

        # counteract the erosion of the lower part of the lungs
        lung_mask_cleaned[:, :, :lower_boundary] = self.morphology.binary_dilation(lung_mask_cleaned[:, :, :lower_boundary], self.morphology.ball(3))

        # close gaps
        lung_mask_cleaned = self.morphology.binary_closing(lung_mask_cleaned, self.morphology.ball(5))

        # Compute distance transform for the entire lung mask
        dist_transform = self.ndi.distance_transform_edt(lung_mask_cleaned)

        # Find local maxima in the distance transform
        lung_markers = self.peak_local_max(dist_transform,
                                    num_peaks=2,  # We want exactly 2 peaks for left and right lung
                                    min_distance=30,  # Minimum pixels between peaks
                                    threshold_rel=0.5)  # Only peaks above 50% of max distance
        
        # Compute the distance transform
        distance = self.ndi.distance_transform_edt(lung_mask_cleaned)
        markers = self.np.zeros_like(lung_mask_cleaned, dtype=self.np.int32)
        for i, marker in enumerate(lung_markers):
            markers[int(marker[0]), int(marker[1]), int(marker[2])] = i + 1

        labels = self.segmentation.watershed(-distance, markers, mask=lung_mask_cleaned)

        unique_labels = self.np.unique(labels)

# Exclude background label (0)
        unique_labels = unique_labels[unique_labels != 0]

        # Create an array of ones with the same shape as labels
        # This will allow us to count the number of pixels per label
        pixel_array = self.np.ones_like(labels, dtype=self.np.int32)

        # Compute the sum of pixels for each label
        sizes = self.ndi.sum(pixel_array, labels, index=unique_labels)

        # Create a dictionary mapping labels to sizes
        label_sizes = dict(zip(unique_labels, sizes))

        # Sort labels by size (from largest to smallest)
        sorted_labels = sorted(label_sizes.items(), key=lambda x: x[1], reverse=True)

        # 'sorted_labels' is a list of tuples: (label, size)

        if len(sorted_labels) >= 2:
            largest_labels = [sorted_labels[0][0], sorted_labels[1][0]]
        else:
            print("Not enough labels found.")
            largest_labels = [label for label, size in sorted_labels]

        # Create a mask for the largest regions
        largest_regions_mask = self.np.isin(labels, largest_labels)

        labels = self.np.where(largest_regions_mask, labels, 0)

        # Get unique labels excluding background (label 0)
        unique_labels = self.np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        # Initialize a dictionary to hold centroids
        centroids = {}

        for label in unique_labels:
            # Create a mask for the current label
            mask = labels == label
            
            # Compute the centroid
            coords = self.np.argwhere(mask)
            centroid = coords.mean(axis=0) # (z, y, x)
            
            # Store the centroid
            centroids[label] = centroid

        # Collect labels and their corresponding x-coordinate of the centroid
        label_x_positions = [(label, centroid[1]) for label, centroid in centroids.items()]

        # Sort labels based on x-coordinate
        # Since we are considering image space, we need to decide if lower x means left or right
        # In radiological convention, the left side of the image is the patient's right side

        # Sort labels based on x-coordinate (from smallest to largest)
        sorted_labels = sorted(label_x_positions, key=lambda x: x[1])

        # Now assign left and right lungs based on position

        # Unpack labels and centroids
        if len(sorted_labels) >= 2:
            # The label with the larger x-coordinate corresponds to the patient's left lung
            right_lung_label = sorted_labels[0][0]  # Smaller x-coordinate
            left_lung_label = sorted_labels[1][0]   # Larger x-coordinate
        else:
            print("Error: Unable to identify left and right lungs.")

        # Create masks for left and right lungs
        left_lung = labels == left_lung_label
        right_lung = labels == right_lung_label

        referenceVolumeArray = slicer.util.arrayFromVolume(referenceVolume)
        left_lung_ref = referenceVolumeArray == 3
        right_lung_ref = referenceVolumeArray == 2
        # body_mask_ref = referenceVolumeArray.get_fdata()

        dice_left = self.compute_dice_coefficient(left_lung, left_lung_ref)
        # left_s = f"Dice Coefficient (Left Lung): {dice_left:.4f}\n"
        dice_right = self.compute_dice_coefficient(right_lung, right_lung_ref)
        # right_s = f"Dice Coefficient (Right Lung): {dice_right:.4f}\n"

        # full_s = left_s + right_s
        # self.outputLabel.setText(full_s)

        seg = self.np.zeros_like(referenceVolumeArray)
        seg = self.np.where(left_lung, 3, seg)
        seg = self.np.where(right_lung, 2, seg)
        seg = self.np.where(airway_mask, 1, seg)

        seg_image = sitk.GetImageFromArray(seg)

        input_image = sitkUtils.PullVolumeFromSlicer(inputVolume)
        seg_image.CopyInformation(input_image)

        sitkUtils.PushVolumeToSlicer(seg_image, outputVolume)

        outputVolume.CreateDefaultDisplayNodes()
        outputVolume.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeRed")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module

        # cliParams = {
        #     "InputVolume": inputVolume.GetID(),
        #     "OutputVolume": outputVolume.GetID(),
        #     "ThresholdValue": imageThreshold,
        #     "ThresholdType": "Above" if invert else "Below",
        # }
        # cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        # slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

        return dice_left, dice_right


#
# SegmentLungsTest
#


class SegmentLungsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SegmentLungs1()

    def test_SegmentLungs1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SegmentLungs1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SegmentLungsLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
