import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map } from 'rxjs/operators';


@Injectable({
  providedIn: 'root'
})
export class ResultService {

  constructor(private http: HttpClient) {}

  getImage() {
     return this.http.get(
        "http://localhost:6001/generate",
        { responseType: "blob" }
      )

  }
}
