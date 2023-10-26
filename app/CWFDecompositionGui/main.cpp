#include "polyscope/pick.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include <igl/boundary_loop.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <filesystem>
#include <iostream>

#include "../../CommonTools.h"
#include "../../LoadSaveIO.h"
#include "../../PaintGeometry.h"
#include "../../json.hpp"

#include "../../Decomposition/CWFDecomposition.h"
#include "../../KnoppelStripePatterns.h"
#include "../../Upsampling/Subdivision.h"


#include <CLI/CLI.hpp>


#include <igl/triangle/triangulate.h>

static void generateShearingCase() {
  Eigen::MatrixXd planeV(4, 2);
  Eigen::MatrixXi planeE(4, 2);
  planeV << 0, 0, 2, 0, 2, 1, 0, 1;
  planeE << 0, 1, 1, 2, 2, 3, 3, 0;

  Eigen::MatrixXd V2d;
  Eigen::MatrixXi F;
  Eigen::MatrixXi H(0, 2);
  const std::string flags = "q20a" + std::to_string(0.005);
  igl::triangle::triangulate(planeV, planeE, H, flags, V2d, F);

  Eigen::MatrixXd triV, shearingV, upTriV, upShearingV;
  Eigen::MatrixXi triF, upF;

  triV.resize(V2d.rows(), 3);
  triV.setZero();
  triV.block(0, 0, triV.rows(), 2) = V2d;
  triF = F;

  shearingV = triV;
  for (int i = 0; i < shearingV.rows(); i++) shearingV(i, 0) = triV(i, 0) + 0.2 * triV(i, 1);

  std::string curDir = std::filesystem::current_path().string();
  std::cout << "curDir: " << curDir << std::endl;
  igl::writeOBJ(curDir + "/restCoarseRect.obj", triV, triF);
  igl::writeOBJ(curDir + "/baseCoarseRect.obj", shearingV, triF);

  // upsampling
  Mesh triMesh;
  triMesh.Populate(triV, triF);

  Mesh upMesh;
  ComplexWrinkleField::Subdivide(triMesh, upMesh, 3, true);

  upTriV = upMesh.GetPos();
  upF = upMesh.GetFace();
  upShearingV = upTriV;

  Eigen::MatrixXd perturb = upShearingV;
  perturb.setRandom();
  perturb.col(0).setZero();
  perturb.col(1).setZero();
  // clamped vertices
  Eigen::VectorXi bnds;
  igl::boundary_loop(upF, bnds);
  for (int i = 0; i < bnds.rows(); i++) {
    perturb(bnds[i], 2) = 0;
  }

  upShearingV = upShearingV + 1e-4 * perturb;

  for (int i = 0; i < upShearingV.rows(); i++) upShearingV(i, 0) = upTriV(i, 0) + 0.2 * upTriV(i, 1);

  igl::writeOBJ(curDir + "/restFineRect.obj", upTriV, upF);
  igl::writeOBJ(curDir + "/baseFineRect.obj", upShearingV, upF);


  std::string path = curDir + "/restFineRect_clamped_vertices.dat";
  std::ofstream ofs(path);
  ofs << bnds.rows() << std::endl;
  ofs << '#' << std::endl;
  for (int i = 0; i < bnds.rows(); i++) {
    ofs << bnds[i] << " " << upShearingV.row(bnds[i]) << std::endl;
  }

  igl::boundary_loop(triF, bnds);
  path = curDir + "/restCoarseRect_clamped_vertices.dat";
  ofs = std::ofstream(path);
  ofs << bnds.rows() << std::endl;
  ofs << '#' << std::endl;
  for (int i = 0; i < bnds.rows(); i++) {
    ofs << bnds[i] << " " << triV.row(bnds[i]) << std::endl;
  }
}

MatrixX triV, upV, refUpV, wrinkledV, refWrinkledV;
Eigen::MatrixXi triF, upF, refUpF, wrinkledF;
Mesh baseMesh, upMesh, refUpMesh;
ComplexWrinkleField::CWF baseCWF, upCWF, baseRefCWF, upRefCWF;

VectorX amp, omega, refAmp, refOmega;
VectorX upAmp, upOmega, upPhase, refUpAmp, refUpOmega, refUpPhase;
VectorX zvals, upZvals, refZvals, refUpZvals;

MatrixX faceOmega, refFaceOmega;
MatrixX upFaceOmega, refUpFaceOmega;

std::string workingFolder = "";
int upsampleTimes = 0;
double wrinkleAmpRatio = 1.0;

float vecratio = 0.01;
bool isFixedBnd = false;
double ampMax = 1;

PaintGeometry mPaint;

int updateViewHelper(const MatrixX& basePos, const Eigen::MatrixXi& baseFaces, const MatrixX& upsampledPos,
                     const Eigen::MatrixXi& upsampledFaces, const MatrixX& wrinkledPos, const VectorX& baseAmplitude,
                     const MatrixX& baseFaceOmega, const VectorX& upsampledAmplitude, const VectorX& upsampledPhase,
                     const MatrixX& upsampledFaceOmega, double shiftx, double shifty, std::string meshSuffix = "_ref",
                     bool isFirstTime = true) {
  int curShift = 0;
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("base mesh" + meshSuffix, basePos, baseFaces);
    polyscope::getSurfaceMesh("base mesh" + meshSuffix)->translate({curShift * shiftx, shifty, 0});
  }

  auto baseOmegaPatterns =
      polyscope::getSurfaceMesh("base mesh" + meshSuffix)
          ->addFaceVectorQuantity("frequency field", vecratio * baseFaceOmega, polyscope::VectorType::AMBIENT);
  baseOmegaPatterns->setEnabled(true);
  auto baseAmpPatterns =
      polyscope::getSurfaceMesh("base mesh" + meshSuffix)->addVertexScalarQuantity("opt amplitude", baseAmplitude);
  baseAmpPatterns->setMapRange({0, ampMax});
  baseAmpPatterns->setColorMap("coolwarm");
  baseAmpPatterns->setEnabled(true);

  curShift++;
  // phase pattern
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("upsampled phase mesh" + meshSuffix, upsampledPos, upsampledFaces);
    polyscope::getSurfaceMesh("upsampled phase mesh" + meshSuffix)->translate({curShift * shiftx, shifty, 0});
  }

  mPaint.SetNormalization(false);
  MatrixX phaseColor = mPaint.PaintPhi(upsampledPhase);
  auto phasePatterns =
      polyscope::getSurfaceMesh("upsampled phase mesh" + meshSuffix)->addVertexColorQuantity("vertex phi", phaseColor);
  phasePatterns->setEnabled(true);

  polyscope::getSurfaceMesh("upsampled phase mesh" + meshSuffix)
      ->addFaceVectorQuantity("subdivided frequency field", vecratio * upsampledFaceOmega,
                              polyscope::VectorType::AMBIENT);
  curShift++;

  // amp pattern
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("upsampled ampliude mesh" + meshSuffix, upsampledPos, upsampledFaces);
    polyscope::getSurfaceMesh("upsampled ampliude mesh" + meshSuffix)->translate({curShift * shiftx, shifty, 0});
  }

  auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude mesh" + meshSuffix)
                         ->addVertexScalarQuantity("vertex amplitude", upsampledAmplitude);
  ampPatterns->setMapRange({0, ampMax});
  ampPatterns->setColorMap("coolwarm");
  ampPatterns->setEnabled(true);

  curShift++;

  // wrinkle mesh
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("wrinkled mesh" + meshSuffix, wrinkledPos, upsampledFaces);
    polyscope::getSurfaceMesh("wrinkled mesh" + meshSuffix)->setSurfaceColor({80 / 255.0, 122 / 255.0, 91 / 255.0});
    polyscope::getSurfaceMesh("wrinkled mesh" + meshSuffix)->translate({curShift * shiftx, shifty, 0});
  }

  else
    polyscope::getSurfaceMesh("wrinkled mesh" + meshSuffix)->updateVertexPositions(wrinkledPos);
  return curShift;
}

void updateView(bool isFirstTime = true) {
  double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
  double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());

  ampMax = std::max(std::max(std::max(refAmp.maxCoeff(), amp.maxCoeff()), refUpAmp.maxCoeff()), upAmp.maxCoeff());

  int curShift = updateViewHelper(triV, triF, refUpV, refUpF, refWrinkledV, refAmp, refFaceOmega, refUpAmp, refUpPhase,
                                  refUpFaceOmega, shiftx, 0, "_ref", isFirstTime);

  curShift = updateViewHelper(triV, triF, upV, upF, wrinkledV, amp, faceOmega, upAmp, upPhase, upFaceOmega, shiftx,
                              shifty, "_opt", isFirstTime);
}

void subdivideMeshHelper(const ComplexWrinkleField::CWF& cwf, ComplexWrinkleField::CWF& upcwf,
                         const bool isFixedBoundary, const int upLevel, const double wrinkleAmpRatio,
                         MatrixX& faceOmega, Mesh& upsampledMesh, MatrixX& upsampledV, Eigen::MatrixXi& upsampledF,
                         MatrixX& wrinkledPos, Eigen::MatrixXi& wrinkledFace, VectorX& upsampledOmega,
                         VectorX& upsampledZvals, MatrixX& upsampledFaceOmega, VectorX& upsampledPhase,
                         VectorX& upsampledAmp) {
  ComplexWrinkleField::Subdivide(cwf, upcwf, upLevel, isFixedBoundary);

  RescaleZvals(upcwf._zvals, upcwf._amp, upsampledZvals);
  upsampledMesh = upcwf._mesh;
  upsampledMesh.GetPos(upsampledV);
  upsampledMesh.GetFace(upsampledF);
  GetWrinkledMesh(upsampledV, upsampledF, upsampledZvals, wrinkledPos, wrinkleAmpRatio, false);
  wrinkledFace = upsampledF;


  faceOmega = IntrinsicEdgeVec2FaceVec(cwf._omega, cwf._mesh);
  upsampledFaceOmega = IntrinsicEdgeVec2FaceVec(upcwf._omega, upcwf._mesh);


  upsampledAmp.setZero(upsampledZvals.size() / 2);
  upsampledPhase.setZero(upsampledZvals.size() / 2);

  for (int i = 0; i < upsampledZvals.size() / 2; i++) {
    std::complex<double> z(upsampledZvals[i], upsampledZvals[i + upsampledZvals.size() / 2]);
    upsampledAmp[i] = std::abs(z);
    upsampledPhase[i] = std::arg(z);
  }
}

void subdivideMesh(bool isSubdivRef = true) {
  if (isSubdivRef)
    subdivideMeshHelper(baseRefCWF, upRefCWF, isFixedBnd, upsampleTimes, wrinkleAmpRatio, refFaceOmega, refUpMesh,
                        refUpV, refUpF, refWrinkledV, refUpF, refUpOmega, refUpZvals, refUpFaceOmega, refUpPhase,
                        refUpAmp);
  else
    subdivideMeshHelper(baseCWF, upCWF, isFixedBnd, upsampleTimes, 1.0, faceOmega, upMesh, upV, upF, wrinkledV, upF,
                        upOmega, upZvals, upFaceOmega, upPhase, upAmp);
}

bool loadProblem(std::string loadFileName = "") {
  if (loadFileName == "") loadFileName = igl::file_dialog_open();

  std::cout << "load file in: " << loadFileName << std::endl;
  using json = nlohmann::json;
  std::ifstream inputJson(loadFileName);
  if (!inputJson) {
    std::cerr << "missing json file in " << loadFileName << std::endl;
    return false;
  }

  std::string filePath = loadFileName;
  std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
  int id = filePath.rfind("/");
  workingFolder = filePath.substr(0, id + 1);
  std::cout << "working folder: " << workingFolder << std::endl;

  json jval;
  inputJson >> jval;

  std::string meshFile = jval["mesh_name"];
  upsampleTimes = jval["upsampled_times"];
  if (upsampleTimes > 2) upsampleTimes = 2;


  meshFile = workingFolder + meshFile;
  igl::readOBJ(meshFile, triV, triF);
  baseMesh.Populate(triV, triF);


  int nedges = baseMesh.GetEdgeCount();
  int nverts = triV.rows();

  std::string initAmpPath = jval["init_amp"];
  std::string initOmegaPath = jval["init_omega"];
  std::string initZValsPath = "zvals.txt";
  if (jval.contains(std::string_view{"init_zvals"})) {
    initZValsPath = jval["init_zvals"];
  }

  if (jval.contains(std::string_view{"wrinkle_amp_ratio"})) {
    wrinkleAmpRatio = jval["wrinkle_amp_ratio"];
  }
  std::cout << "wrinkle amplitude scaling ratio: " << wrinkleAmpRatio << std::endl;

  if (!LoadEdgeOmega(workingFolder + initOmegaPath, nedges, omega)) {
    std::cout << "missing init edge omega file." << std::endl;
    return false;
  }

  // convert old stored edge omega to the current order
  omega = SwapEdgeVec(triF, omega, 0);
  std::cout << "convert finished, omega size: " << omega.rows() << std::endl;

  if (!LoadVertexZvals(workingFolder + initZValsPath, triV.rows(), zvals)) {
    std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
    if (!LoadVertexAmp(workingFolder + initAmpPath, triV.rows(), amp)) {
      std::cout << "missing init amp file: " << std::endl;
      return false;
    }

    else {
      VectorX edgeArea, vertArea;
      edgeArea = GetEdgeArea(baseMesh);
      vertArea = GetVertArea(baseMesh);
      RoundZvalsFromEdgeOmegaVertexMag(baseMesh, omega, amp, edgeArea, vertArea, nverts, zvals);
    }
  } else {
    amp.setZero(triV.rows());
    for (int i = 0; i < zvals.size() / 2; i++) {
      amp(i) = std::sqrt(zvals[i] * zvals[i] + zvals[i + nverts] * zvals[i + nverts]);
    }
  }


  refAmp = amp * 2;
  refOmega = omega / 2;
  refZvals = zvals;

  for (auto& z : refZvals) z *= 2.0;

  VectorX edgeArea, vertArea;
  edgeArea = GetEdgeArea(baseMesh);
  vertArea = GetVertArea(baseMesh);
  RoundZvalsFromEdgeOmegaVertexMag(baseMesh, refOmega, refAmp, edgeArea, vertArea, nverts, refZvals);

  amp = refAmp;
  omega = refOmega;
  baseCWF = ComplexWrinkleField::CWF(amp, omega, NormalizeZvals(zvals), baseMesh);
  baseRefCWF = ComplexWrinkleField::CWF(refAmp, refOmega, NormalizeZvals(refZvals), baseMesh);


  std::cout << "start to subdivide" << std::endl;
  subdivideMesh(true);  // reference mesh
  subdivideMesh(false); // initial mesh
  std::cout << "subdivide done, start to update view" << std::endl;
  updateView();
  return true;
}

void callback() {
  ImGui::PushItemWidth(100);
  float w = ImGui::GetContentRegionAvailWidth();
  float p = ImGui::GetStyle().FramePadding.x;
  if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0))) {
    loadProblem();
  }
  ImGui::SameLine(0, p);
  if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0))) {
  }
  if (ImGui::CollapsingHeader("Wrinkle Mesh Upsampling Options", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::InputInt("upsampled level", &upsampleTimes)) {
      if (upsampleTimes >= 0) {
        subdivideMesh(true);  // reference mesh
        subdivideMesh(false); // initial mesh
        updateView(true);
      }
    }
    if (ImGui::Checkbox("fix bnd", &isFixedBnd)) {
      subdivideMesh(true);  // reference mesh
      subdivideMesh(false); // initial mesh
      updateView(true);
    }
  }

  if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::InputDouble("wrinkle amp scaling ratio", &wrinkleAmpRatio)) {
      if (wrinkleAmpRatio >= 0) {
        GetWrinkledMesh(refUpV, refUpF, refUpZvals, refWrinkledV, wrinkleAmpRatio, false);
        updateView();
      }
    }
    if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1)) {
      updateView(false);
    }
  }
  if (ImGui::Button("CWF Projection", ImVec2(-1, 0))) {
    Mesh refWrinkledMesh = refUpMesh;
    refWrinkledMesh.SetPos(refWrinkledV);

    ComplexWrinkleField::CWFDecomposition decompModel(refWrinkledMesh);
    decompModel.Initialization(baseCWF, upsampleTimes);

    MatrixX pos;
    baseCWF._mesh.GetPos(pos);

    
    decompModel.OptimizePhase();
    // decompModel.precomputationForPhase();
    decompModel.OptimizeBasemesh();
    decompModel.GetCWF(baseCWF);

    VectorX unitZvals = baseCWF._zvals;
    omega = baseCWF._omega;
    amp = baseCWF._amp;
    RescaleZvals(unitZvals, amp, zvals);
    subdivideMesh(false);
    updateView(false);
    
  }

  ImGui::PopItemWidth();
}

int main(int argc, char** argv) {
  std::string inputFile = "";
  CLI::App app("Complex Wrinkle Field Decomposition");
  app.add_option("input,-i,--input", inputFile, "Input model")->check(CLI::ExistingFile);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();
  loadProblem(inputFile);

  polyscope::view::upDir = polyscope::view::UpDir::ZUp;

  // Add the callback
  polyscope::state::userCallback = callback;

  polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

  // Show the gui
  polyscope::show();

  return 0;
}