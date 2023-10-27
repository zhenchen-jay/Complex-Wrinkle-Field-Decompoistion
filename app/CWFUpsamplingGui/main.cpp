#include "polyscope/pick.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <iostream>

#include "../../CWF.h"
#include "../../CommonTools.h"
#include "../../LoadSaveIO.h"
#include "../../PaintGeometry.h"
#include "../../json.hpp"

#include "../../KnoppelStripePatterns.h"
#include "../../Upsampling/Subdivision.h"
#include "../../Upsampling/BaseLoop.h"
#include "../../Upsampling/ComplexLoop.h"
#include "../../Upsampling/StandardLoop.h"

#include <CLI/CLI.hpp>

MatrixX triV, upV, wrinkledV, wrinkledV1, compositeWrinkledV;
Eigen::MatrixXi triF, upF, wrinkledF;
Mesh baseMesh, upMesh;

ComplexWrinkleField::CWF baseCWF, upCWF, baseCWF1, upCWF1;

VectorX amp, omega, amp1, omega1;
VectorX upAmp, upOmega, upPhase, upAmp1, upOmega1, upPhase1;
VectorX zvals, upZvals, zvals1, upZvals1;

MatrixX faceOmega, faceOmega1;
MatrixX upFaceOmega, upFaceOmega1;

std::string workingFolder = "";
int upsampleTimes = 0;
double wrinkleAmpRatio = 1.0;
double secFrequencyRatio = 0;
double secAmpRatio = 0;

float vecratio = 0.1;
bool isFixedBnd = false;

PaintGeometry mPaint;

int updateViewHelper(const MatrixX& basePos, const Eigen::MatrixXi& baseFaces, const MatrixX& upsampledPos,
                     const Eigen::MatrixXi& upsampledFaces, const MatrixX& wrinkledPos, const VectorX& baseAmplitude,
                     const MatrixX& baseFaceOmega, const VectorX& upsampledAmplitude, const VectorX& upsampledPhase,
                     const MatrixX& upsampledFaceOmega, double shiftx, double shifty, int meshId = 0,
                     bool isFirstTime = true) {
  int curShift = 0;
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("base mesh" + std::to_string(meshId), basePos, baseFaces);
    polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))->translate({curShift * shiftx, shifty, 0});
  }

  polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))
      ->addFaceVectorQuantity("frequency field", vecratio * baseFaceOmega, polyscope::VectorType::AMBIENT);
  auto baseAmpPatterns = polyscope::getSurfaceMesh("base mesh" + std::to_string(meshId))
                             ->addVertexScalarQuantity("opt amplitude", baseAmplitude);
  baseAmpPatterns->setEnabled(true);

  curShift++;
  // phase pattern
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("upsampled phase mesh" + std::to_string(meshId), upsampledPos, upsampledFaces);
    polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))
        ->translate({curShift * shiftx, shifty, 0});
  }

  mPaint.SetNormalization(false);
  MatrixX phaseColor = mPaint.PaintPhi(upsampledPhase);
  auto phasePatterns = polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))
                           ->addVertexColorQuantity("vertex phi", phaseColor);
  phasePatterns->setEnabled(true);

  polyscope::getSurfaceMesh("upsampled phase mesh" + std::to_string(meshId))
      ->addFaceVectorQuantity("subdivided frequency field", vecratio * upsampledFaceOmega,
                              polyscope::VectorType::AMBIENT);
  curShift++;

  // amp pattern
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId), upsampledPos, upsampledFaces);
    polyscope::getSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId))
        ->translate({curShift * shiftx, shifty, 0});
  }

  auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude mesh" + std::to_string(meshId))
                         ->addVertexScalarQuantity("vertex amplitude", upsampledAmplitude);
  ampPatterns->setEnabled(true);

  curShift++;

  // wrinkle mesh
  if (isFirstTime) {
    polyscope::registerSurfaceMesh("wrinkled mesh" + std::to_string(meshId), wrinkledPos, upsampledFaces);
    polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))
        ->setSurfaceColor({80 / 255.0, 122 / 255.0, 91 / 255.0});
    polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))->translate({curShift * shiftx, shifty, 0});
  }

  else
    polyscope::getSurfaceMesh("wrinkled mesh" + std::to_string(meshId))->updateVertexPositions(wrinkledPos);
  return curShift;
}

void updateView(bool isFirstTime = true) {
  double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
  double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());

  int curShift = updateViewHelper(triV, triF, upV, upF, wrinkledV, amp, faceOmega, upAmp, upPhase, upFaceOmega, shiftx,
                                  0, 0, isFirstTime);
  if (secAmpRatio && secFrequencyRatio) {
    updateViewHelper(triV, triF, upV, upF, wrinkledV1, amp1, faceOmega1, upAmp1, upPhase1, upFaceOmega1, shiftx, shifty,
                     isFirstTime);

    // wrinkle mesh
    if (isFirstTime) {
      polyscope::registerSurfaceMesh("composite wrinkled mesh", compositeWrinkledV, upF);
      polyscope::getSurfaceMesh("composite wrinkled mesh")->setSurfaceColor({80 / 255.0, 122 / 255.0, 91 / 255.0});
      polyscope::getSurfaceMesh("composite wrinkled mesh")->translate({curShift * shiftx, 2 * shifty, 0});
    }

    else
      polyscope::getSurfaceMesh("composite wrinkled mesh")->updateVertexPositions(compositeWrinkledV);
  }
}

void subdivideMesh() {
  // upsampleTimes = 1;
  ComplexWrinkleField::Subdivide(baseCWF, upCWF, upsampleTimes, isFixedBnd);

  upMesh = upCWF._mesh;
  upMesh.GetPos(upV);
  upMesh.GetFace(upF);

  RescaleZvals(upCWF._zvals, upCWF._amp, upZvals);
  GetWrinkledMesh(upV, upF, upZvals, wrinkledV, wrinkleAmpRatio, false);
  wrinkledF = upF;
  upOmega = upCWF._omega;

  faceOmega = IntrinsicEdgeVec2FaceVec(omega, baseMesh);
  upFaceOmega = IntrinsicEdgeVec2FaceVec(upOmega, upMesh);

  upAmp.setZero(upZvals.size() / 2);
  upPhase.setZero(upZvals.size() / 2);

  for (int i = 0; i < upZvals.size() / 2; i++) {
    std::complex<double> z(upZvals[i], upZvals[i + upZvals.size() / 2]);
    upAmp[i] = std::abs(z);
    upPhase[i] = std::arg(z);
  }
  std::cout << "first wave compute done!" << std::endl;

  if (secFrequencyRatio && secAmpRatio) {
    omega1 = secFrequencyRatio * omega;
    amp1 = secAmpRatio * amp;

    VectorX edgeArea, vertArea;
    edgeArea = GetEdgeArea(baseMesh);
    vertArea = GetVertArea(baseMesh);
    RoundZvalsFromEdgeOmegaVertexMag(baseMesh, omega1, amp1, edgeArea, vertArea, amp1.rows(), zvals1);
    baseCWF1 = ComplexWrinkleField::CWF(amp1, omega1, NormalizeZvals(zvals1), baseMesh);

    ComplexWrinkleField::Subdivide(baseCWF1, upCWF1, upsampleTimes, isFixedBnd);
    RescaleZvals(upCWF1._zvals, upCWF1._amp, upZvals1);
    std::cout << "second subdivision done" << std::endl;

    MatrixX upN;
    igl::per_vertex_normals(upV, upF, upN);
    wrinkledV1 = upV;
    compositeWrinkledV = upV;
    for (int i = 0; i < wrinkledV.rows(); i++) {
      compositeWrinkledV.row(i) = upV.row(i) + upN.row(i) * (upZvals[i] + upZvals1[i]) * wrinkleAmpRatio;
      wrinkledV1.row(i) = upV.row(i) + upN.row(i) * (upZvals1[i]) * wrinkleAmpRatio;
    }

    upOmega1 = upCWF1._omega;

    faceOmega1 = IntrinsicEdgeVec2FaceVec(omega1, baseMesh);
    upFaceOmega1 = IntrinsicEdgeVec2FaceVec(upOmega1, upMesh);

    upAmp1.setZero(upZvals1.size() / 2);
    upPhase1.setZero(upZvals1.size() / 2);

    for (int i = 0; i < upZvals1.size(); i++) {
      std::complex<double> z(upZvals1[i], upZvals1[i + upZvals.size() / 2]);
      upAmp1[i] = std::abs(z);
      upPhase1[i] = std::arg(z);
    }
    std::cout << "secondary wave compute done!" << std::endl;
  }
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
    assert(zvals.size() == 2 * nverts);
    for (int i = 0; i < nverts; i++) {
      amp(i) = std::sqrt(zvals[i] * zvals[i] + zvals[i + nverts] * zvals[i + nverts]);
    }
  }
  std::cout << "start to subdivide" << std::endl;
  baseCWF = ComplexWrinkleField::CWF(amp, omega, NormalizeZvals(zvals), baseMesh);

  subdivideMesh();
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
    std::string savePath = igl::file_dialog_save();
    igl::writeOBJ(savePath, wrinkledV, wrinkledF);

    if (secFrequencyRatio && secAmpRatio) {
      savePath = igl::file_dialog_save();
      igl::writeOBJ(savePath, wrinkledV1, wrinkledF);

      savePath = igl::file_dialog_save();
      igl::writeOBJ(savePath, compositeWrinkledV, wrinkledF);
    }
  }
  if (ImGui::CollapsingHeader("Wrinkle Mesh Upsampling Options", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::InputInt("upsampled level", &upsampleTimes)) {
      if (upsampleTimes >= 0) {
        subdivideMesh();
        updateView(true);
      }
    }
    if (ImGui::Checkbox("fix bnd", &isFixedBnd)) {
      subdivideMesh();
      updateView(true);
    }
  }

  if (ImGui::CollapsingHeader("Second wave options", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::InputDouble("frequency ratio", &secFrequencyRatio)) {
      if (secFrequencyRatio < 0) secFrequencyRatio = 0;
    }
    if (ImGui::InputDouble("Amplitude ratio", &secAmpRatio)) {
      if (secAmpRatio < 0) secAmpRatio = 0;
    }
    if (ImGui::Button("wrinkles on wrinkles")) {
      subdivideMesh();
      updateView();
    }
  }

  if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::InputDouble("wrinkle amp scaling ratio", &wrinkleAmpRatio)) {
      if (wrinkleAmpRatio >= 0) {
        GetWrinkledMesh(upV, upF, upZvals, wrinkledV, wrinkleAmpRatio, false);
        updateView();
      }
    }
    if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1)) {
      updateView(false);
    }
  }

  if (ImGui::Button("Test Loop")) {
      int nverts = baseMesh.GetVertCount();
      int nedges = baseMesh.GetEdgeCount();

      VectorX randVec = VectorX::Random(nverts);
      VectorX dRandVec = VectorX::Zero(nedges);

      SparseMatrixX d0(nedges, nverts);
      std::vector<TripletX> T;

      for (int i = 0; i < nedges; i++) {
          int v0 = baseMesh.GetEdgeVerts(i)[0];
          int v1 = baseMesh.GetEdgeVerts(i)[1];
          T.push_back({ i, v0, 1.0 });
          T.push_back({ i, v1, -1.0 });
          dRandVec[i] = randVec[v0] - randVec[v1];
      }
      d0.setFromTriplets(T.begin(), T.end());

      std::unique_ptr<ComplexWrinkleField::BaseLoop> loopPtr = std::make_unique<ComplexWrinkleField::StandardLoop>();
      loopPtr->SetMesh(baseMesh);
      loopPtr->SetBndFixFlag(isFixedBnd);

      SparseMatrixX S0, S1;
      loopPtr->BuildS0(S0);
      loopPtr->BuildS1(S1);

      VectorX subdVec = S1 * dRandVec;
      VectorX subVec = S0 * randVec;

      std::vector<std::vector<int>> edgeToVert;
      loopPtr->GetSubdividedEdges(edgeToVert);

      T.clear();
      SparseMatrixX d1(edgeToVert.size(), S0.rows());
      VectorX subdVec1 = VectorX::Zero(edgeToVert.size());
      for (int i = 0; i < edgeToVert.size(); i++) {
          int v0 = edgeToVert[i][0];
          int v1 = edgeToVert[i][1];
          T.push_back({ i, v0, 1.0 });
          T.push_back({ i, v1, -1.0 });
          subdVec1[i] = subVec[v0] - subVec[v1];
      }
      d1.setFromTriplets(T.begin(), T.end());

      std::cout << "d * S0 - S1 * d: " << (d1 * S0 - S1 * d0).norm() << std::endl;

      std::cout << "S0 norm: " << S0.norm() << std::endl;
      std::cout << "S1 norm: " << S1.norm() << std::endl;

      MatrixX diff = (d1 * S0 - S1 * d0).toDense();
      for (int i = 0; i < diff.rows(); i++) {
          if (diff.row(i).norm() > 1e-4) {
              std::cout << "edge id: " << i << " is wrong" << std::endl;
          }
      }

      //std::cout << (subdVec - subdVec1).transpose() << std::endl;
      
  }

  ImGui::PopItemWidth();
}


int main(int argc, char** argv) {
  std::string inputFile = "";
  CLI::App app("Wrinkle Upsampling");
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
