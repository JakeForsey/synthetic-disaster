import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map } from 'rxjs/operators';

interface Result {
  result: string;
}


@Injectable({
  providedIn: 'root'
})
export class ResultService {

  constructor(private http: HttpClient) { }

  getResult() {
      return this.http.get<Result>(
      "http://localhost:6001/generate"
    ).pipe(map((response: Result) => response.result))
  }
}
