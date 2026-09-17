import segmentationMode from "@ohif/mode-segmentation";

const id = "@monailabel/mode";
const extensionDependencies = {
  ...segmentationMode.extensionDependencies,
  "@monailabel/extension": "0.1.0",
};
export default {
  id,
  extensionDependencies,
  modeFactory(options) {
    const mode = segmentationMode.modeFactory(options);
    return {
      ...mode,
      id,
      routeName: "monailabel",
      displayName: "MONAI Label · Annotation",
      extensions: extensionDependencies,
      routes: mode.routes.map((route) => ({
        ...route,
        layoutTemplate(context) {
          const layout = route.layoutTemplate(context);
          return {
            ...layout,
            props: {
              ...layout.props,
              rightPanels: ["@monailabel/extension.panelModule.assistant"],
              rightPanelResizable: true,
              leftPanelClosed: window.innerWidth < 1200,
              rightPanelClosed: window.innerWidth < 700,
              rightPanelInitialExpandedWidth: Math.min(
                400,
                window.innerWidth - 80,
              ),
              rightPanelMinimumExpandedWidth: Math.min(
                300,
                window.innerWidth - 80,
              ),
            },
          };
        },
      })),
    };
  },
};
